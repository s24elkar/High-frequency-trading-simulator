#include "order_flow/OrderFlow.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

#include "error.hpp"

namespace order_flow {

void EventStream::reserve(std::size_t n) {
    events_.reserve(n);
}

void EventStream::add(double time, double mark) {
    if (!std::isfinite(time) || time < 0.0) {
        HFT_THROW(std::invalid_argument("EventStream::add received non-finite or negative time"));
    }
    if (!std::isfinite(mark)) {
        HFT_THROW(std::invalid_argument("EventStream::add received non-finite mark"));
    }
    if (!events_.empty() && time < events_.back().time) {
        std::ostringstream oss;
        oss << "event time " << time << " precedes last time " << events_.back().time;
        HFT_THROW(std::invalid_argument(oss.str()));
    }
    events_.push_back(Event{time, mark});
}

void EventStream::clear() noexcept {
    events_.clear();
}

std::size_t EventStream::size() const noexcept {
    return events_.size();
}

bool EventStream::empty() const noexcept {
    return events_.empty();
}

double EventStream::last_time() const noexcept {
    return events_.empty() ? 0.0 : events_.back().time;
}

const EventStream::Container& EventStream::data() const noexcept {
    return events_;
}

EventStream::iterator EventStream::begin() noexcept {
    return events_.begin();
}

EventStream::iterator EventStream::end() noexcept {
    return events_.end();
}

EventStream::const_iterator EventStream::begin() const noexcept {
    return events_.begin();
}

EventStream::const_iterator EventStream::end() const noexcept {
    return events_.end();
}

std::vector<double> EventStream::interarrival_times() const {
    std::vector<double> deltas;
    if (events_.size() < 2) {
        return deltas;
    }
    deltas.reserve(events_.size() - 1);
    for (std::size_t i = 1; i < events_.size(); ++i) {
        deltas.push_back(events_[i].time - events_[i - 1].time);
    }
    return deltas;
}

double IntensityFunction::reproduction_mean(double mark_expectation) const {
    (void)mark_expectation;
    return 0.0;
}

PoissonIntensity::PoissonIntensity(double mu) : mu_(mu) {
    if (!(mu_ > 0.0)) {
        HFT_THROW(std::invalid_argument("PoissonIntensity requires mu > 0"));
    }
}

double PoissonIntensity::value(double, const EventStream&) const {
    return mu_;
}

ExponentialKernel::ExponentialKernel(double alpha, double beta)
    : alpha_(alpha), beta_(beta) {
    if (!(alpha_ >= 0.0)) {
        HFT_THROW(std::invalid_argument("ExponentialKernel requires alpha >= 0"));
    }
    if (!(beta_ > 0.0)) {
        HFT_THROW(std::invalid_argument("ExponentialKernel requires beta > 0"));
    }
}

double ExponentialKernel::evaluate(double lag, double mark) const {
    if (lag < 0.0) {
        return 0.0;
    }
    return alpha_ * mark * std::exp(-beta_ * lag);
}

double ExponentialKernel::integral(double mark_expectation) const {
    return alpha_ * mark_expectation / beta_;
}

double ExponentialKernel::decay(double state, double dt) const {
    if (dt <= 0.0) {
        return state;
    }
    return state * std::exp(-beta_ * dt);
}

double ExponentialKernel::jump(double mark) const {
    return alpha_ * mark;
}

double ExponentialKernel::intensity(double mu, double state) const noexcept {
    const double lambda = mu + state;
    return lambda > 0.0 ? lambda : 0.0;
}

double ExponentialKernel::alpha() const noexcept {
    return alpha_;
}

double ExponentialKernel::beta() const noexcept {
    return beta_;
}

PowerLawKernel::PowerLawKernel(double alpha, double c, double gamma)
    : alpha_(alpha), c_(c), gamma_(gamma) {
    if (!(alpha_ >= 0.0)) {
        HFT_THROW(std::invalid_argument("PowerLawKernel requires alpha >= 0"));
    }
    if (!(c_ > 0.0)) {
        HFT_THROW(std::invalid_argument("PowerLawKernel requires c > 0"));
    }
    if (!(gamma_ > 1.0)) {
        HFT_THROW(std::invalid_argument("PowerLawKernel requires gamma > 1"));
    }
}

double PowerLawKernel::evaluate(double lag, double mark) const {
    if (lag < 0.0) {
        return 0.0;
    }
    return alpha_ * mark * std::pow(lag + c_, -gamma_);
}

double PowerLawKernel::integral(double mark_expectation) const {
    const double exponent = 1.0 - gamma_;
    return alpha_ * mark_expectation * std::pow(c_, exponent) / (gamma_ - 1.0);
}

double PowerLawKernel::alpha() const noexcept {
    return alpha_;
}

double PowerLawKernel::c() const noexcept {
    return c_;
}

double PowerLawKernel::gamma() const noexcept {
    return gamma_;
}

CustomKernel::CustomKernel(EvaluateFn evaluator, IntegralFn integral)
    : evaluator_(std::move(evaluator)), integral_(std::move(integral)) {
    if (!evaluator_) {
        HFT_THROW(std::invalid_argument("CustomKernel requires a valid evaluator function"));
    }
    if (!integral_) {
        HFT_THROW(std::invalid_argument("CustomKernel requires a valid integral function"));
    }
}

double CustomKernel::evaluate(double lag, double mark) const {
    if (lag < 0.0) {
        return 0.0;
    }
    return evaluator_(lag, mark);
}

double CustomKernel::integral(double mark_expectation) const {
    return integral_(mark_expectation);
}

HawkesIntensity::HawkesIntensity(double mu, std::shared_ptr<const HawkesKernel> kernel)
    : mu_(mu), kernel_(std::move(kernel)) {
    if (!kernel_) {
        HFT_THROW(std::invalid_argument("HawkesIntensity requires a kernel"));
    }
    if (!(mu_ >= 0.0)) {
        HFT_THROW(std::invalid_argument("HawkesIntensity requires mu >= 0"));
    }
}

double HawkesIntensity::value(double t, const EventStream& history) const {
    double lambda = mu_;
    for (const auto& event : history) {
        const double lag = t - event.time;
        if (lag < 0.0) {
            continue;
        }
        lambda += kernel_->evaluate(lag, event.mark);
    }
    return std::max(lambda, 0.0);
}

double HawkesIntensity::reproduction_mean(double mark_expectation) const {
    return kernel_->integral(mark_expectation);
}

double HawkesIntensity::mu() const noexcept {
    return mu_;
}

const HawkesKernel& HawkesIntensity::kernel() const noexcept {
    return *kernel_;
}

PoissonProcess::PoissonProcess(double mu) : mu_(mu) {
    if (!(mu_ > 0.0)) {
        HFT_THROW(std::invalid_argument("PoissonProcess requires mu > 0"));
    }
}

EventStream PoissonProcess::simulate(double horizon, std::uint64_t seed) const {
    MarkSampler sampler = [](std::mt19937_64&) { return 1.0; };
    return simulate(horizon, sampler, seed);
}

EventStream PoissonProcess::simulate(double horizon, MarkSampler sampler, std::uint64_t seed) const {
    if (!sampler) {
        HFT_THROW(std::invalid_argument("PoissonProcess::simulate requires a valid mark sampler"));
    }
    if (!(horizon >= 0.0)) {
        HFT_THROW(std::invalid_argument("PoissonProcess::simulate requires horizon >= 0"));
    }

    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> expo(mu_);

    EventStream stream;
    double time = 0.0;
    while (time < horizon) {
        time += expo(rng);
        if (time > horizon) {
            break;
        }
        stream.add(time, sampler(rng));
    }
    return stream;
}

PoissonProcess::InterarrivalSummary PoissonProcess::evaluate_interarrivals(const EventStream& stream, double mu) {
    InterarrivalSummary summary;
    if (stream.size() < 2) {
        summary.theoretical_mean = 1.0 / mu;
        summary.samples = stream.size() > 0 ? stream.size() - 1 : 0;
        summary.empirical_mean = std::numeric_limits<double>::quiet_NaN();
        summary.absolute_error = std::numeric_limits<double>::quiet_NaN();
        return summary;
    }

    const auto deltas = stream.interarrival_times();
    const double empirical = std::accumulate(deltas.begin(), deltas.end(), 0.0) / static_cast<double>(deltas.size());
    const double theoretical = 1.0 / mu;

    summary.empirical_mean = empirical;
    summary.theoretical_mean = theoretical;
    summary.absolute_error = std::abs(empirical - theoretical);
    summary.samples = deltas.size();
    return summary;
}

HawkesProcess::HawkesProcess(double mu, std::shared_ptr<const HawkesKernel> kernel, double mark_expectation)
    : mu_(mu),
      kernel_(std::move(kernel)),
      mark_expectation_(mark_expectation),
      sampler_([mark_expectation](std::mt19937_64&) { return mark_expectation; }),
      exp_kernel_(nullptr) {
    if (!kernel_) {
        HFT_THROW(std::invalid_argument("HawkesProcess requires a kernel"));
    }
    if (!(mu_ >= 0.0)) {
        HFT_THROW(std::invalid_argument("HawkesProcess requires mu >= 0"));
    }
    if (!(mark_expectation_ > 0.0)) {
        HFT_THROW(std::invalid_argument("HawkesProcess requires mark expectation > 0"));
    }
    if (kernel_->is_exponential()) {
        exp_kernel_ = static_cast<const ExponentialKernel*>(kernel_.get());
    }
}

void HawkesProcess::set_mark_sampler(MarkSampler sampler) {
    if (!sampler) {
        HFT_THROW(std::invalid_argument("HawkesProcess::set_mark_sampler requires a valid sampler"));
    }
    sampler_ = std::move(sampler);
}

void HawkesProcess::set_mark_expectation(double value) {
    if (!(value > 0.0)) {
        HFT_THROW(std::invalid_argument("HawkesProcess::set_mark_expectation requires value > 0"));
    }
    mark_expectation_ = value;
}

EventStream HawkesProcess::simulate(double horizon, std::uint64_t seed) const {
    if (!(horizon >= 0.0)) {
        HFT_THROW(std::invalid_argument("HawkesProcess::simulate requires horizon >= 0"));
    }
    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> expo(1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    EventStream stream;
    SimulationState state;

    auto simulate_exp_kernel = [&]() {
        while (state.time < horizon) {
            const double lambda_star = compute_intensity(state.time, stream, state);
            if (lambda_star <= 0.0) {
                break;
            }
            const double wait = schedule_next_event(lambda_star, rng, expo);
            const double candidate_time = state.time + wait;
            if (candidate_time > horizon) {
                break;
            }

            SimulationState candidate_state = state;
            candidate_state.time = candidate_time;
            decay_state(candidate_state, wait);

            const double lambda_candidate = compute_intensity(candidate_time, stream, candidate_state);
            const double accept_prob = std::clamp(lambda_candidate / lambda_star, 0.0, 1.0);
            if (unif(rng) <= accept_prob) {
                const double mark = sampler_ ? sampler_(rng) : mark_expectation_;
                register_event(candidate_state, 0.0, mark);
                stream.add(candidate_time, mark);
                state = candidate_state;
            } else {
                state = candidate_state;
            }
        }
    };

    auto simulate_generic_kernel = [&]() {
        while (state.time < horizon) {
            const double lambda_star = compute_intensity(state.time, stream, state);
            if (lambda_star <= 0.0) {
                break;
            }
            const double wait = schedule_next_event(lambda_star, rng, expo);
            const double candidate_time = state.time + wait;
            if (candidate_time > horizon) {
                break;
            }

            SimulationState candidate_state = state;
            candidate_state.time = candidate_time;

            const double lambda_candidate = compute_intensity(candidate_time, stream, candidate_state);
            const double accept_prob = std::clamp(lambda_candidate / lambda_star, 0.0, 1.0);

            if (unif(rng) <= accept_prob) {
                const double mark = sampler_ ? sampler_(rng) : mark_expectation_;
                stream.add(candidate_time, mark);
                state = candidate_state;
            } else {
                state = candidate_state;
            }
        }
    };

    if (exp_kernel_) {
        simulate_exp_kernel();
    } else {
        simulate_generic_kernel();
    }

    return stream;
}

double HawkesProcess::branching_ratio() const {
    return kernel_->integral(mark_expectation_);
}

double HawkesProcess::spectral_radius() const {
    return branching_ratio();
}

HawkesProcess::StabilityReport HawkesProcess::check_stability(double tolerance) const {
    if (!(tolerance >= 0.0)) {
        HFT_THROW(std::invalid_argument("HawkesProcess::check_stability requires tolerance >= 0"));
    }
    const double rho = branching_ratio();
    const double spectral = spectral_radius();
    return StabilityReport{
        rho,
        spectral,
        tolerance,
        rho < 1.0 - tolerance
    };
}

double HawkesProcess::mu() const noexcept {
    return mu_;
}

const HawkesKernel& HawkesProcess::kernel() const noexcept {
    return *kernel_;
}

double HawkesProcess::mark_expectation() const noexcept {
    return mark_expectation_;
}

double HawkesProcess::evaluate_kernel(double lag, double mark) const {
    if (!kernel_) {
        return 0.0;
    }
    return kernel_->evaluate(lag, mark);
}

double HawkesProcess::compute_intensity(double t, const EventStream& history, const SimulationState& state) const {
    if (exp_kernel_) {
        return exp_kernel_->intensity(mu_, state.excitation);
    }
    double lambda = mu_;
    for (const auto& event : history) {
        const double lag = t - event.time;
        if (lag < 0.0) {
            continue;
        }
        lambda += evaluate_kernel(lag, event.mark);
    }
    return lambda > 0.0 ? lambda : 0.0;
}

double HawkesProcess::schedule_next_event(double lambda_star, std::mt19937_64& rng,
                                          std::exponential_distribution<double>& expo) const {
    expo.param(std::exponential_distribution<double>::param_type(lambda_star));
    return expo(rng);
}

void HawkesProcess::decay_state(SimulationState& state, double dt) const {
    if (!exp_kernel_ || dt <= 0.0) {
        return;
    }
    state.excitation = exp_kernel_->decay(state.excitation, dt);
}

void HawkesProcess::register_event(SimulationState& state, double dt, double mark) const {
    if (!exp_kernel_) {
        (void)dt;
        (void)mark;
        return;
    }
    if (dt > 0.0) {
        decay_state(state, dt);
    }
    state.excitation += exp_kernel_->jump(mark);
}

} // namespace order_flow
