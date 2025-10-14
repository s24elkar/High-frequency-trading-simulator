#include <catch2/catch_test_macros.hpp>

#include "order_flow/HawkesMLE.hpp"
#include "order_flow/OrderFlow.hpp"

#include <cmath>
#include <memory>

namespace {

using order_flow::EventStream;
using order_flow::ExponentialKernel;
using order_flow::HawkesProcess;
using order_flow::calibration::HawkesParameters;
using order_flow::calibration::HawkesLikelihoodResult;
using order_flow::calibration::HawkesMLEConfig;

double finite_difference_derivative(
    const EventStream& events,
    double horizon,
    HawkesParameters params,
    int index) {
    constexpr double kMinStep = 1e-6;
    double step = 0.0;
    switch (index) {
        case 0: {
            step = std::max(kMinStep, params.mu * 1e-4);
            step = std::min(step, params.mu * 0.25);
            HawkesParameters plus = params;
            HawkesParameters minus = params;
            plus.mu += step;
            minus.mu = std::max(params.mu - step, 1e-8);
            const auto f_plus = order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, plus);
            const auto f_minus = order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, minus);
            return (f_plus.log_likelihood - f_minus.log_likelihood) / (plus.mu - minus.mu);
        }
        case 1: {
            step = std::max(kMinStep, params.alpha * 1e-4);
            step = std::min(step, std::max(params.alpha * 0.25, 1e-5));
            HawkesParameters plus = params;
            HawkesParameters minus = params;
            plus.alpha += step;
            minus.alpha = std::max(params.alpha - step, 1e-8);
            const auto f_plus = order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, plus);
            const auto f_minus = order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, minus);
            return (f_plus.log_likelihood - f_minus.log_likelihood) / (plus.alpha - minus.alpha);
        }
        case 2: {
            step = std::max(kMinStep, params.beta * 1e-4);
            step = std::min(step, params.beta * 0.25);
            HawkesParameters plus = params;
            HawkesParameters minus = params;
            plus.beta += step;
            minus.beta = std::max(params.beta - step, 1e-8);
            const auto f_plus = order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, plus);
            const auto f_minus = order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, minus);
            return (f_plus.log_likelihood - f_minus.log_likelihood) / (plus.beta - minus.beta);
        }
        default:
            return 0.0;
    }
}

} // namespace

TEST_CASE("hawkes_mle_gradient_matches_finite_difference") {
    EventStream events;
    events.add(0.05, 1.0);
    events.add(0.21, 1.0);
    events.add(0.40, 1.0);
    events.add(0.95, 1.0);
    events.add(1.40, 1.0);
    events.add(1.75, 1.0);

    const double horizon = 2.0;
    const HawkesParameters params{0.8, 0.4, 1.3};
    const HawkesLikelihoodResult analytic =
        order_flow::calibration::exponential_hawkes_loglikelihood(events, horizon, params);
    REQUIRE(analytic.valid);

    const double fd_mu = finite_difference_derivative(events, horizon, params, 0);
    const double fd_alpha = finite_difference_derivative(events, horizon, params, 1);
    const double fd_beta = finite_difference_derivative(events, horizon, params, 2);

    CHECK(std::fabs(analytic.gradient[0] - fd_mu) <= 1e-6);
    CHECK(std::fabs(analytic.gradient[1] - fd_alpha) <= 1e-6);
    CHECK(std::fabs(analytic.gradient[2] - fd_beta) <= 1e-6);
}

TEST_CASE("hawkes_mle_recovers_synthetic_parameters") {
    const double true_mu = 0.7;
    const double true_alpha = 0.5;
    const double true_beta = 1.4;
    const double horizon = 200.0;

    auto kernel = std::make_shared<ExponentialKernel>(true_alpha, true_beta);
    HawkesProcess process(true_mu, kernel);
    process.set_mark_expectation(1.0);

    const EventStream events = process.simulate(horizon, 1234);
    REQUIRE(events.size() > 10);

    auto run_fit = [&](bool enforce_stationarity) {
        HawkesMLEConfig config;
        config.gradient_tolerance = 5e-5;
        config.parameter_tolerance = 1e-5;
        config.max_iterations = 200;
        config.max_branching_ratio = 0.99;
        config.enforce_stationarity = enforce_stationarity;

        const auto result = order_flow::calibration::fit_exponential_hawkes_mle(events, horizon, config);

        REQUIRE(result.converged);
        CHECK(result.gradient_norm < 5e-5);
        CHECK(std::fabs(result.params.mu - true_mu) < 0.05);
        CHECK(std::fabs(result.params.alpha - true_alpha) < 0.06);
        CHECK(std::fabs(result.params.beta - true_beta) < 0.1);
    };

    run_fit(false);
    run_fit(true);
}
