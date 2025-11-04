#include "hawkes_engine.hpp"

#include "error.hpp"
#include "perf/Profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#define HFT_PREFETCH(addr) __builtin_prefetch(addr, 0, 1)
#else
#define HFT_PREFETCH(addr) (void)0
#endif

namespace simulator {

namespace {

constexpr double kEpsilon = 1e-12;

std::vector<double> flatten_matrix(const ExponentialHawkesProcess::Matrix& matrix) {
    std::vector<double> flat;
    if (matrix.empty()) {
        return flat;
    }
    const std::size_t rows = matrix.size();
    const std::size_t cols = matrix.front().size();
    flat.resize(rows * cols);
    double* dest = flat.data();
    for (const auto& row : matrix) {
        dest = std::copy(row.begin(), row.end(), dest);
    }
    return flat;
}

} // namespace

ExponentialHawkesProcess::ExponentialHawkesProcess(std::vector<double> mu,
                                                   Matrix alpha,
                                                   Matrix beta,
                                                   MarkSampler mark_sampler)
    : mu_(std::move(mu)),
      mark_sampler_(mark_sampler ? std::move(mark_sampler)
                                 : MarkSampler([](std::mt19937_64&) { return 1.0; })),
      time_(0.0) {
    dim_ = mu_.size();
    validate_parameters(alpha, beta);
    alpha_flat_ = flatten_matrix(alpha);
    beta_flat_ = flatten_matrix(beta);
    excitation_.assign(dim_ * dim_, 0.0);
    candidate_excitation_.assign(dim_ * dim_, 0.0);
    lambda_ = mu_;
    candidate_lambda_.assign(dim_, 0.0);
}

void ExponentialHawkesProcess::validate_parameters(const Matrix& alpha,
                                                   const Matrix& beta) const {
    const std::size_t d = dim_;
    if (d == 0) {
        HFT_THROW(std::invalid_argument("ExponentialHawkesProcess requires non-empty mu vector"));
    }
    const auto check_matrix = [&](const Matrix& M, const char* label) {
        if (M.size() != d) {
            HFT_THROW(std::invalid_argument(std::string(label) + " matrix has incompatible row count"));
        }
        for (const auto& row : M) {
            if (row.size() != d) {
                HFT_THROW(std::invalid_argument(std::string(label) + " matrix has incompatible column count"));
            }
        }
    };
    check_matrix(alpha, "alpha");
    check_matrix(beta, "beta");

    for (double mu_value : mu_) {
        if (!std::isfinite(mu_value) || mu_value < 0.0) {
            HFT_THROW(std::invalid_argument("mu components must be finite and non-negative"));
        }
    }
    for (const auto& row : alpha) {
        for (double v : row) {
            if (!std::isfinite(v) || v < 0.0) {
                HFT_THROW(std::invalid_argument("alpha coefficients must be finite and non-negative"));
            }
        }
    }
    for (const auto& row : beta) {
        for (double v : row) {
            if (!std::isfinite(v) || v <= 0.0) {
                HFT_THROW(std::invalid_argument("beta coefficients must be finite and strictly positive"));
            }
        }
    }
}

std::size_t ExponentialHawkesProcess::dimension() const noexcept {
    return dim_;
}

double ExponentialHawkesProcess::current_time() const noexcept {
    return time_;
}

const std::vector<double>& ExponentialHawkesProcess::intensities() const noexcept {
    return lambda_;
}

void ExponentialHawkesProcess::reset(double start_time) {
    time_ = start_time;
    excitation_.assign(dim_ * dim_, 0.0);
    candidate_excitation_.assign(dim_ * dim_, 0.0);
    lambda_ = mu_;
    candidate_lambda_.assign(dim_, 0.0);
}

double ExponentialHawkesProcess::total_intensity() const noexcept {
    double sum = 0.0;
    for (double v : lambda_) {
        sum += v;
    }
    return sum;
}

void ExponentialHawkesProcess::decay_state(double dt) {
    HFT_PROFILE_SCOPE("Hawkes::decay_state");
    if (dt <= 0.0) {
        return;
    }
    const std::size_t d = dim_;
    double* excitation_ptr = excitation_.data();
    double* lambda_ptr = lambda_.data();
    const double* mu_ptr = mu_.data();
    const double* beta_ptr = beta_flat_.data();

    for (std::size_t i = 0; i < d; ++i) {
        double lambda_i = mu_ptr[i];
        double* row_excitation = excitation_ptr + i * d;
        const double* row_beta = beta_ptr + i * d;
#if defined(__GNUC__) || defined(__clang__)
        if (i + 1 < d) {
            HFT_PREFETCH(excitation_ptr + (i + 1) * d);
            HFT_PREFETCH(beta_ptr + (i + 1) * d);
        }
#endif
        for (std::size_t j = 0; j < d; ++j) {
#if defined(__GNUC__) || defined(__clang__)
            if (j + 4 < d) {
                HFT_PREFETCH(row_excitation + j + 4);
                HFT_PREFETCH(row_beta + j + 4);
            }
#endif
            double state = row_excitation[j];
            if (state > kEpsilon || state < -kEpsilon) {
                state *= std::exp(-row_beta[j] * dt);
                row_excitation[j] = state;
            }
            lambda_i += row_excitation[j];
        }
        lambda_ptr[i] = (lambda_i > 0.0) ? lambda_i : 0.0;
    }
}

std::size_t ExponentialHawkesProcess::draw_dimension(double lambda_sum,
                                                     std::mt19937_64& rng) const {
    HFT_PROFILE_SCOPE("Hawkes::draw_dimension");
    if (!(lambda_sum > 0.0)) {
        HFT_THROW(std::logic_error("draw_dimension called with non-positive intensity sum"));
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double threshold = unif(rng) * lambda_sum;
    double cumulative = 0.0;
    for (std::size_t i = 0; i < lambda_.size(); ++i) {
        cumulative += lambda_[i];
        if (threshold <= cumulative || i + 1 == lambda_.size()) {
            return i;
        }
    }
    return lambda_.size() - 1;
}

HawkesEvent ExponentialHawkesProcess::sample_next(std::mt19937_64& rng) {
    HFT_PROFILE_SCOPE("Hawkes::sample_next");
    const std::size_t d = dim_;
    std::exponential_distribution<double> expo;
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double lambda_star = total_intensity();
    if (!(lambda_star > 0.0)) {
        HFT_THROW(std::runtime_error("total intensity is non-positive; cannot sample next event"));
    }

    const std::size_t matrix_size = d * d;
    while (true) {
        std::memcpy(candidate_excitation_.data(), excitation_.data(), matrix_size * sizeof(double));
        expo.param(std::exponential_distribution<double>::param_type(lambda_star));
        const double wait = expo(rng);
        const double candidate_time = time_ + wait;

        double lambda_sum_candidate = 0.0;
        double* candidate_exc_ptr = candidate_excitation_.data();
        double* candidate_lambda_ptr = candidate_lambda_.data();
        const double* mu_ptr = mu_.data();
        const double* beta_ptr = beta_flat_.data();

        for (std::size_t i = 0; i < d; ++i) {
            double lambda_i = mu_ptr[i];
            double* row_excitation = candidate_exc_ptr + i * d;
            const double* row_beta = beta_ptr + i * d;
#if defined(__GNUC__) || defined(__clang__)
            if (i + 1 < d) {
                HFT_PREFETCH(candidate_exc_ptr + (i + 1) * d);
                HFT_PREFETCH(beta_ptr + (i + 1) * d);
            }
#endif
            for (std::size_t j = 0; j < d; ++j) {
#if defined(__GNUC__) || defined(__clang__)
                if (j + 4 < d) {
                    HFT_PREFETCH(row_excitation + j + 4);
                    HFT_PREFETCH(row_beta + j + 4);
                }
#endif
                double state = row_excitation[j];
                if (state > kEpsilon || state < -kEpsilon) {
                    state *= std::exp(-row_beta[j] * wait);
                    row_excitation[j] = state;
                    lambda_i += state;
                } else {
                    lambda_i += state;
                }
            }
            if (lambda_i < 0.0) {
                lambda_i = 0.0;
            }
            candidate_lambda_ptr[i] = lambda_i;
            lambda_sum_candidate += lambda_i;
        }

        const double acceptance = lambda_sum_candidate / lambda_star;
        if (lambda_sum_candidate <= 0.0) {
            lambda_star = std::max(lambda_sum_candidate, kEpsilon);
            time_ = candidate_time;
            excitation_.swap(candidate_excitation_);
            lambda_.swap(candidate_lambda_);
            continue;
        }

        if (unif(rng) <= acceptance) {
            // Accept candidate event
            lambda_.swap(candidate_lambda_);
            excitation_.swap(candidate_excitation_);
            time_ = candidate_time;

            const std::size_t dim = draw_dimension(lambda_sum_candidate, rng);
            const double mark = mark_sampler_(rng);

            double post_sum = 0.0;
            double* excitation_ptr = excitation_.data();
            double* lambda_ptr = lambda_.data();
            const double* alpha_ptr = alpha_flat_.data();
            for (std::size_t i = 0; i < d; ++i) {
                const std::size_t offset = i * d + dim;
#if defined(__GNUC__) || defined(__clang__)
                if (i + 1 < d) {
                    HFT_PREFETCH(excitation_ptr + (i + 1) * d + dim);
                    HFT_PREFETCH(alpha_ptr + (i + 1) * d + dim);
                }
#endif
                const double jump = alpha_ptr[offset] * mark;
                const double updated_excitation = excitation_ptr[offset] + jump;
                excitation_ptr[offset] = updated_excitation;
                double updated_lambda = lambda_ptr[i] + jump;
                if (updated_lambda < 0.0) {
                    updated_lambda = 0.0;
                }
                lambda_ptr[i] = updated_lambda;
                post_sum += updated_lambda;
            }

            return HawkesEvent{
                time_,
                dim,
                post_sum,
                lambda_ptr[dim]
            };
        }

        // Reject candidate; advance time and states without registering event
        lambda_star = lambda_sum_candidate;
        time_ = candidate_time;
        excitation_.swap(candidate_excitation_);
        lambda_.swap(candidate_lambda_);
        if (lambda_star <= kEpsilon) {
            lambda_star = kEpsilon;
        }
    }
}

} // namespace simulator

#undef HFT_PREFETCH
