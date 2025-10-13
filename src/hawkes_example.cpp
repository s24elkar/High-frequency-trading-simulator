#include "order_flow/OrderFlow.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

int main() {
    using namespace order_flow;

    auto exp_kernel = std::make_shared<ExponentialKernel>(0.8, 1.2);
    HawkesProcess exp_process(0.2, exp_kernel);
    exp_process.set_mark_sampler([](std::mt19937_64& rng) {
        std::lognormal_distribution<double> d(0.0, 0.5);
        return d(rng);
    });
    auto exp_stream = exp_process.simulate(200.0, 123);
    std::cout << "[EXP] generated " << exp_stream.size() << " events\n";

    auto pl_kernel = std::make_shared<PowerLawKernel>(0.1, 0.1, 1.4);
    HawkesProcess pl_process(0.2, pl_kernel);
    pl_process.set_mark_sampler([](std::mt19937_64& rng) {
        std::exponential_distribution<double> d(1.0);
        return d(rng);
    });
    auto pl_stream = pl_process.simulate(50.0, 456);
    std::cout << "[PL ] generated " << pl_stream.size() << " events\n";

    PoissonProcess poisson(0.3);
    auto poisson_stream = poisson.simulate(50.0, 789);
    auto summary = PoissonProcess::evaluate_interarrivals(poisson_stream, 0.3);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[POI] generated " << poisson_stream.size()
              << " events, empirical Δt mean = " << summary.empirical_mean
              << ", theoretical 1/mu = " << summary.theoretical_mean << '\n';
}
