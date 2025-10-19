#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace simulator {

struct HawkesEvent {
    double time;
    std::size_t dimension;
    double intensity_total;
    double intensity_dimension;
};

class IHawkesProcess {
public:
    virtual ~IHawkesProcess() = default;
    virtual std::size_t dimension() const noexcept = 0;
    virtual double current_time() const noexcept = 0;
    virtual const std::vector<double>& intensities() const noexcept = 0;
    virtual HawkesEvent sample_next(std::mt19937_64& rng) = 0;
    virtual void reset(double start_time = 0.0) = 0;
};

class ExponentialHawkesProcess : public IHawkesProcess {
public:
    using Matrix = std::vector<std::vector<double>>;
    using MarkSampler = std::function<double(std::mt19937_64&)>;

    ExponentialHawkesProcess(std::vector<double> mu,
                             Matrix alpha,
                             Matrix beta,
                             MarkSampler mark_sampler = {});

    std::size_t dimension() const noexcept override;
    double current_time() const noexcept override;
    const std::vector<double>& intensities() const noexcept override;
    HawkesEvent sample_next(std::mt19937_64& rng) override;
    void reset(double start_time = 0.0) override;

private:
    void validate_parameters() const;
    void decay_state(double dt);
    double total_intensity() const noexcept;
    std::size_t draw_dimension(double lambda_sum, std::mt19937_64& rng) const;

    std::vector<double> mu_;
    Matrix alpha_;
    Matrix beta_;
    MarkSampler mark_sampler_;

    double time_;
    Matrix excitation_;
    std::vector<double> lambda_;
};

} // namespace simulator

