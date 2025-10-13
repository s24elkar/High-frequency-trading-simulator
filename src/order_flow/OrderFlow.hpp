#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace order_flow {

struct Event {
    double time{};
    double mark{1.0};
};

class EventStream {
public:
    using Container = std::vector<Event>;
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;

    EventStream() = default;

    void reserve(std::size_t n);
    void add(double time, double mark = 1.0);
    void clear() noexcept;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] double last_time() const noexcept;
    [[nodiscard]] const Container& data() const noexcept;

    iterator begin() noexcept;
    iterator end() noexcept;
    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;

    [[nodiscard]] std::vector<double> interarrival_times() const;

private:
    Container events_;
};

class IntensityFunction {
public:
    virtual ~IntensityFunction() = default;
    virtual double value(double t, const EventStream& history) const = 0;
    virtual double reproduction_mean(double mark_expectation) const;
};

class PoissonIntensity final : public IntensityFunction {
public:
    explicit PoissonIntensity(double mu);
    double value(double t, const EventStream& history) const override;
private:
    double mu_;
};

class HawkesKernel {
public:
    virtual ~HawkesKernel() = default;
    virtual double evaluate(double lag, double mark) const = 0;
    virtual double integral(double mark_expectation) const = 0;
};

class ExponentialKernel final : public HawkesKernel {
public:
    ExponentialKernel(double alpha, double beta);
    double evaluate(double lag, double mark) const override;
    double integral(double mark_expectation) const override;
    double decay(double state, double dt) const;
    double jump(double mark) const;
    double intensity(double mu, double state) const noexcept;

    [[nodiscard]] double alpha() const noexcept;
    [[nodiscard]] double beta() const noexcept;

private:
    double alpha_;
    double beta_;
};

class PowerLawKernel final : public HawkesKernel {
public:
    PowerLawKernel(double alpha, double c, double gamma);
    double evaluate(double lag, double mark) const override;
    double integral(double mark_expectation) const override;

    [[nodiscard]] double alpha() const noexcept;
    [[nodiscard]] double c() const noexcept;
    [[nodiscard]] double gamma() const noexcept;

private:
    double alpha_;
    double c_;
    double gamma_;
};

class CustomKernel final : public HawkesKernel {
public:
    using EvaluateFn = std::function<double(double, double)>;
    using IntegralFn = std::function<double(double)>;

    CustomKernel(EvaluateFn evaluator, IntegralFn integral);
    double evaluate(double lag, double mark) const override;
    double integral(double mark_expectation) const override;

private:
    EvaluateFn evaluator_;
    IntegralFn integral_;
};

class HawkesIntensity final : public IntensityFunction {
public:
    HawkesIntensity(double mu, std::shared_ptr<const HawkesKernel> kernel);
    double value(double t, const EventStream& history) const override;
    double reproduction_mean(double mark_expectation) const override;

    [[nodiscard]] double mu() const noexcept;
    [[nodiscard]] const HawkesKernel& kernel() const noexcept;

private:
    double mu_;
    std::shared_ptr<const HawkesKernel> kernel_;
};

class PoissonProcess {
public:
    using MarkSampler = std::function<double(std::mt19937_64&)>;

    explicit PoissonProcess(double mu);

    EventStream simulate(double horizon, std::uint64_t seed = 0) const;
    EventStream simulate(double horizon, MarkSampler sampler, std::uint64_t seed = 0) const;

    struct InterarrivalSummary {
        double empirical_mean{};
        double theoretical_mean{};
        double absolute_error{};
        std::size_t samples{};
    };

    static InterarrivalSummary evaluate_interarrivals(const EventStream& stream, double mu);

private:
    double mu_;
};

class HawkesProcess {
public:
    using MarkSampler = std::function<double(std::mt19937_64&)>;

    HawkesProcess(double mu, std::shared_ptr<const HawkesKernel> kernel, double mark_expectation = 1.0);

    void set_mark_sampler(MarkSampler sampler);
    void set_mark_expectation(double value);

    EventStream simulate(double horizon, std::uint64_t seed = 42) const;

    [[nodiscard]] double branching_ratio() const;
    [[nodiscard]] double spectral_radius() const;

    struct StabilityReport {
        double branching_ratio{};
        double spectral_radius{};
        double tolerance{};
        bool stable{};
    };

    [[nodiscard]] StabilityReport check_stability(double tolerance = 1e-6) const;

    [[nodiscard]] double mu() const noexcept;
    [[nodiscard]] const HawkesKernel& kernel() const noexcept;
    [[nodiscard]] double mark_expectation() const noexcept;

private:
    struct SimulationState {
        double time{0.0};
        double excitation{0.0};
    };

    [[nodiscard]] double evaluate_kernel(double lag, double mark) const;
    [[nodiscard]] double compute_intensity(double t, const EventStream& history, const SimulationState& state) const;
    [[nodiscard]] double schedule_next_event(double lambda_star, std::mt19937_64& rng, std::exponential_distribution<double>& expo) const;
    void decay_state(SimulationState& state, double dt) const;
    void register_event(SimulationState& state, double dt, double mark) const;

    double mu_;
    std::shared_ptr<const HawkesKernel> kernel_;
    double mark_expectation_;
    MarkSampler sampler_;
    const ExponentialKernel* exp_kernel_;
};

} // namespace order_flow
