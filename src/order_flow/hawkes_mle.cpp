#include "order_flow/HawkesMLE.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace order_flow::calibration {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
using Vector3 = std::array<double, 3>;
using Matrix3 = std::array<double, 9>;

double log_sum_exp(double a, double b) {
    if (a <= kNegInf) {
        return b;
    }
    if (b <= kNegInf) {
        return a;
    }
    const double m = std::max(a, b);
    if (!std::isfinite(m)) {
        return m;
    }
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double dot(const Vector3& lhs, const Vector3& rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

Vector3 add(const Vector3& lhs, const Vector3& rhs) {
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

Vector3 subtract(const Vector3& lhs, const Vector3& rhs) {
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Vector3 scale(const Vector3& v, double s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

Matrix3 identity_matrix() {
    Matrix3 mat{};
    mat[0] = mat[4] = mat[8] = 1.0;
    return mat;
}

Matrix3 outer(const Vector3& a, const Vector3& b) {
    return {
        a[0] * b[0], a[0] * b[1], a[0] * b[2],
        a[1] * b[0], a[1] * b[1], a[1] * b[2],
        a[2] * b[0], a[2] * b[1], a[2] * b[2]};
}

Matrix3 matrix_add(const Matrix3& lhs, const Matrix3& rhs) {
    Matrix3 result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

Matrix3 matrix_scale(const Matrix3& mat, double s) {
    Matrix3 result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = mat[i] * s;
    }
    return result;
}

Matrix3 matrix_multiply(const Matrix3& lhs, const Matrix3& rhs) {
    Matrix3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int k = 0; k < 3; ++k) {
                value += lhs[row * 3 + k] * rhs[k * 3 + col];
            }
            result[row * 3 + col] = value;
        }
    }
    return result;
}

Vector3 mat_vec(const Matrix3& mat, const Vector3& vec) {
    return {
        mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
        mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
        mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2]};
}

double vector_norm(const Vector3& v) {
    return std::sqrt(dot(v, v));
}

double max_abs_difference(const HawkesParameters& lhs, const HawkesParameters& rhs) {
    const double d_mu = std::abs(lhs.mu - rhs.mu);
    const double d_alpha = std::abs(lhs.alpha - rhs.alpha);
    const double d_beta = std::abs(lhs.beta - rhs.beta);
    return std::max({d_mu, d_alpha, d_beta});
}

HawkesParameters default_initial_guess(const EventStream& events, double horizon, double max_branching_ratio) {
    const auto& data = events.data();
    const std::size_t n = data.size();

    const double safe_horizon = std::max(horizon, 1e-3);
    double mu0 = n > 0 ? static_cast<double>(n) / safe_horizon : 1e-3;
    mu0 = std::max(mu0, 1e-4);

    double alpha0 = 0.5 * mu0;
    alpha0 = std::max(alpha0, 1e-4);

    double beta0 = 1.0;

    const double rho_limit = std::max(max_branching_ratio, 1e-3);
    if (beta0 > 0.0) {
        const double rho = alpha0 / beta0;
        if (rho >= rho_limit) {
            alpha0 = 0.5 * rho_limit * beta0;
        }
    }

    return HawkesParameters{mu0, alpha0, beta0};
}

struct Evaluation {
    bool valid{false};
    double log_likelihood{kNegInf};
    HawkesParameters params{0.0, 0.0, 0.0};
    Vector3 gradient_params{};
    Vector3 gradient_log{};
    double branching_ratio{0.0};
};

Evaluation evaluate_theta(
    const EventStream& events,
    double horizon,
    const HawkesMLEConfig& config,
    const Vector3& theta) {
    Evaluation eval{};

    constexpr double max_exponent = 700.0; // guard exp overflow
    for (double value : theta) {
        if (!std::isfinite(value) || value > max_exponent) {
            return eval;
        }
    }

    const HawkesParameters params{std::exp(theta[0]), std::exp(theta[1]), std::exp(theta[2])};
    if (!std::isfinite(params.mu) || !std::isfinite(params.alpha) || !std::isfinite(params.beta)) {
        return eval;
    }

    auto likelihood = exponential_hawkes_loglikelihood(
        events,
        horizon,
        params,
        config.enforce_stationarity,
        config.max_branching_ratio);
    if (!likelihood.valid) {
        return eval;
    }

    eval.valid = true;
    eval.log_likelihood = likelihood.log_likelihood;
    eval.params = params;
    eval.gradient_params = likelihood.gradient;
    eval.gradient_log = {
        likelihood.gradient[0] * params.mu,
        likelihood.gradient[1] * params.alpha,
        likelihood.gradient[2] * params.beta};
    eval.branching_ratio = likelihood.branching_ratio;
    return eval;
}

void bfgs_update(Matrix3& H, const Vector3& s, const Vector3& y) {
    const double ys = dot(y, s);
    if (ys <= 1e-12) {
        H = identity_matrix();
        return;
    }
    const double rho = 1.0 / ys;
    const Matrix3 identity = identity_matrix();
    const Matrix3 syT = outer(s, y);
    const Matrix3 ysT = outer(y, s);
    const Matrix3 I_minus_rho_syT = matrix_add(identity, matrix_scale(syT, -rho));
    const Matrix3 I_minus_rho_ysT = matrix_add(identity, matrix_scale(ysT, -rho));
    Matrix3 updated = matrix_multiply(I_minus_rho_syT, matrix_multiply(H, I_minus_rho_ysT));
    const Matrix3 ssT = outer(s, s);
    H = matrix_add(updated, matrix_scale(ssT, rho));
}

} // namespace

HawkesLikelihoodResult exponential_hawkes_loglikelihood(
    const EventStream& events,
    double horizon,
    const HawkesParameters& params,
    bool enforce_stationarity,
    double max_branching_ratio) {
    HawkesLikelihoodResult result;

    if (!(std::isfinite(horizon) && horizon >= 0.0)) {
        return result;
    }
    if (!(std::isfinite(params.mu) && params.mu > 0.0)) {
        return result;
    }
    if (!(std::isfinite(params.beta) && params.beta > 0.0)) {
        return result;
    }
    if (!(std::isfinite(params.alpha) && params.alpha >= 0.0)) {
        return result;
    }

    const auto& data = events.data();
    if (!data.empty()) {
        const double last_time = data.back().time;
        if (!std::isfinite(last_time) || last_time < 0.0) {
            return result;
        }
        if (horizon + 1e-12 < last_time) {
            return result;
        }
    }

    const double branching = branching_ratio(params);
    result.branching_ratio = branching;
    if (enforce_stationarity && branching >= max_branching_ratio) {
        return result;
    }

    double log_likelihood = 0.0;
    double grad_mu = 0.0;
    double grad_alpha = 0.0;
    double grad_beta = 0.0;

    const double log_mu = std::log(params.mu);

    double prev_time = 0.0;
    double prev_mark = 0.0;
    double K_prev = 0.0;
    double G_prev = 0.0;
    bool has_prev = false;

    for (const auto& event : data) {
        if (!std::isfinite(event.time) || !std::isfinite(event.mark)) {
            return result;
        }
        if (event.time < 0.0) {
            return result;
        }
        if (has_prev && event.time + 1e-12 < prev_time) {
            return result;
        }
        if (event.mark < 0.0) {
            return result;
        }

        double K_i = 0.0;
        double G_i = 0.0;
        if (has_prev) {
            const double dt = event.time - prev_time;
            const double exponent = -params.beta * dt;
            if (!std::isfinite(exponent)) {
                return result;
            }
            const double decay = std::exp(exponent);
            const double base = K_prev + prev_mark;
            K_i = decay * base;
            G_i = decay * (G_prev - dt * base);
        }

        double log_term = kNegInf;
        if (params.alpha > 0.0 && K_i > 0.0) {
            log_term = std::log(params.alpha) + std::log(K_i);
        }
        const double log_lambda = log_sum_exp(log_mu, log_term);
        if (!std::isfinite(log_lambda)) {
            return result;
        }
        const double inv_lambda = std::exp(-log_lambda);

        log_likelihood += log_lambda;
        grad_mu += inv_lambda;
        grad_alpha += K_i * inv_lambda;
        grad_beta += params.alpha * G_i * inv_lambda;

        prev_time = event.time;
        prev_mark = event.mark;
        K_prev = K_i;
        G_prev = G_i;
        has_prev = true;
    }

    double tail_sum = 0.0;
    double tail_grad = 0.0;
    for (const auto& event : data) {
        const double delta = horizon - event.time;
        if (delta < -1e-12) {
            return result;
        }
        const double exponent = -params.beta * std::max(delta, 0.0);
        if (!std::isfinite(exponent)) {
            return result;
        }
        const double exp_term = std::exp(exponent);
        const double one_minus = -std::expm1(exponent); // 1 - exp(-beta * delta)
        tail_sum += event.mark * one_minus;
        tail_grad += event.mark * std::max(delta, 0.0) * exp_term;
    }

    log_likelihood -= params.mu * horizon;
    if (params.beta > 0.0) {
        log_likelihood -= (params.alpha / params.beta) * tail_sum;
    }

    grad_mu -= horizon;
    if (params.beta > 0.0) {
        grad_alpha -= tail_sum / params.beta;
        grad_beta -= params.alpha * (params.beta * tail_grad - tail_sum) / (params.beta * params.beta);
    }

    result.log_likelihood = log_likelihood;
    result.gradient = {grad_mu, grad_alpha, grad_beta};
    result.valid = true;
    return result;
}

HawkesMLEResult fit_exponential_hawkes_mle(
    const EventStream& events,
    double horizon,
    const HawkesMLEConfig& config) {
    if (!(std::isfinite(horizon) && horizon >= 0.0)) {
        throw std::invalid_argument("horizon must be non-negative and finite");
    }

    double resolved_horizon = horizon;
    if (resolved_horizon <= 0.0) {
        resolved_horizon = events.empty() ? 0.0 : events.data().back().time;
    }
    if (!events.empty() && resolved_horizon + 1e-12 < events.data().back().time) {
        throw std::invalid_argument("horizon must be greater than or equal to the last event time");
    }

    HawkesParameters initial{};
    if (config.initial_guess.has_value()) {
        initial = *config.initial_guess;
    } else {
        initial = default_initial_guess(events, resolved_horizon, config.max_branching_ratio);
    }

    if (!(initial.mu > 0.0)) {
        initial.mu = 1e-3;
    }
    if (!(initial.alpha > 0.0)) {
        initial.alpha = std::min(1e-3, 0.5 * initial.mu);
    }
    if (!(initial.beta > 0.0)) {
        initial.beta = 1.0;
    }

    const Vector3 theta_initial{
        std::log(initial.mu),
        std::log(initial.alpha),
        std::log(initial.beta)};

    Matrix3 H = identity_matrix();
    HawkesMLEResult result;
    result.trajectory.reserve(config.max_iterations + 1);

    Evaluation current = evaluate_theta(events, resolved_horizon, config, theta_initial);
    if (!current.valid) {
        throw std::runtime_error("Initial parameters yield invalid likelihood");
    }

    Vector3 theta = theta_initial;
    result.params = current.params;
    result.log_likelihood = current.log_likelihood;
    result.gradient_norm = vector_norm(current.gradient_params);
    result.trajectory.push_back(current.params);

    std::size_t completed_iters = 0;
    while (completed_iters < config.max_iterations) {
        if (result.gradient_norm <= config.gradient_tolerance) {
            result.converged = true;
            break;
        }

        Vector3 direction = mat_vec(H, current.gradient_log);
        if (dot(direction, current.gradient_log) <= 0.0) {
            H = identity_matrix();
            direction = current.gradient_log;
        }

        double step = 1.0;
        Evaluation candidate{};
        Vector3 theta_candidate{};
        bool accepted = false;
        for (std::size_t ls = 0; ls < config.max_line_search_steps; ++ls) {
            theta_candidate = add(theta, scale(direction, step));
            candidate = evaluate_theta(events, resolved_horizon, config, theta_candidate);
            if (!candidate.valid) {
                step *= config.backtracking_shrink;
                continue;
            }
            const double sufficient = current.log_likelihood + config.wolfe_c1 * step * dot(current.gradient_log, direction);
            if (candidate.log_likelihood < sufficient) {
                step *= config.backtracking_shrink;
                continue;
            }
            accepted = true;
            break;
        }

        if (!accepted) {
            break;
        }

        const Vector3 s = subtract(theta_candidate, theta);
        const Vector3 y = subtract(candidate.gradient_log, current.gradient_log);
        bfgs_update(H, s, y);

        theta = theta_candidate;
        current = candidate;
        result.params = current.params;
        result.log_likelihood = current.log_likelihood;
        result.gradient_norm = vector_norm(current.gradient_params);
        result.trajectory.push_back(current.params);

        ++completed_iters;

        if (result.gradient_norm <= config.gradient_tolerance &&
            max_abs_difference(result.trajectory[result.trajectory.size() - 1],
                               result.trajectory[result.trajectory.size() - 2]) <= config.parameter_tolerance) {
            result.converged = true;
            break;
        }
    }

    result.iterations = completed_iters;
    return result;
}

} // namespace order_flow::calibration
