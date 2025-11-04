#include "order_flow/OrderFlow.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <utility>

#include "error.hpp"

namespace {

using HawkesMarkCallback = double (*)(void*);

thread_local std::string g_last_error;

struct CallbackSampler {
    HawkesMarkCallback fn;
    void* ctx;
    std::exponential_distribution<double> fallback{1.0};

    template <class RNG>
    double operator()(RNG& rng) {
        if (fn) {
            return fn(ctx);
        }
        return fallback(rng);
    }
};

inline std::size_t copy_stream(const order_flow::EventStream& stream, double** times_out, double** marks_out) {
    const std::size_t n = stream.size();
    double* times = nullptr;
    double* marks = nullptr;
    if (n > 0) {
        times = static_cast<double*>(std::malloc(n * sizeof(double)));
        marks = static_cast<double*>(std::malloc(n * sizeof(double)));
        if (!times || !marks) {
            std::free(times);
            std::free(marks);
            HFT_THROW(std::bad_alloc());
        }
        const auto& data = stream.data();
        for (std::size_t i = 0; i < n; ++i) {
            times[i] = data[i].time;
            marks[i] = data[i].mark;
        }
    }
    *times_out = times;
    *marks_out = marks;
    return n;
}

inline std::size_t handle_failure(double** times_out, double** marks_out, const std::string& message) {
    g_last_error = message;
    if (times_out) {
        *times_out = nullptr;
    }
    if (marks_out) {
        *marks_out = nullptr;
    }
    return static_cast<std::size_t>(-1);
}

}  // namespace

extern "C" {

std::size_t hawkes_simulate_exp(double mu, double alpha, double beta, double horizon,
                                std::uint64_t seed, HawkesMarkCallback sampler,
                                void* sampler_ctx, double** times_out, double** marks_out) {
    if (!times_out || !marks_out) {
        return handle_failure(times_out, marks_out, "null output pointer");
    }
    if (!(mu >= 0.0) || !(alpha >= 0.0) || !(beta > 0.0) || !(horizon >= 0.0)) {
        return handle_failure(times_out, marks_out, "invalid exponential Hawkes parameters");
    }

    g_last_error.clear();
    auto kernel = std::shared_ptr<order_flow::ExponentialKernel>(
        new (std::nothrow) order_flow::ExponentialKernel(alpha, beta));
    if (!kernel) {
        return handle_failure(times_out, marks_out, "allocation failure");
    }

    order_flow::HawkesProcess process(mu, kernel);
    CallbackSampler cb{sampler, sampler_ctx};
    if (sampler) {
        process.set_mark_sampler(
            [cb](std::mt19937_64& rng) mutable { return cb(rng); }
        );
    }
    auto result = process.simulate(horizon, seed);
    return copy_stream(result, times_out, marks_out);
}

std::size_t hawkes_simulate_powerlaw(double mu, double alpha, double c, double gamma,
                                      double horizon, std::uint64_t seed, HawkesMarkCallback sampler,
                                      void* sampler_ctx, double** times_out, double** marks_out) {
    if (!times_out || !marks_out) {
        return handle_failure(times_out, marks_out, "null output pointer");
    }
    if (!(mu >= 0.0) || !(alpha >= 0.0) || !(c > 0.0) || !(gamma > 1.0) || !(horizon >= 0.0)) {
        return handle_failure(times_out, marks_out, "invalid power-law Hawkes parameters");
    }

    g_last_error.clear();
    auto kernel = std::shared_ptr<order_flow::PowerLawKernel>(
        new (std::nothrow) order_flow::PowerLawKernel(alpha, c, gamma));
    if (!kernel) {
        return handle_failure(times_out, marks_out, "allocation failure");
    }

    order_flow::HawkesProcess process(mu, kernel);
    CallbackSampler cb{sampler, sampler_ctx};
    if (sampler) {
        process.set_mark_sampler(
            [cb](std::mt19937_64& rng) mutable { return cb(rng); }
        );
    }
    auto result = process.simulate(horizon, seed);
    return copy_stream(result, times_out, marks_out);
}

std::size_t hawkes_simulate_poisson(double mu, double horizon, std::uint64_t seed,
                                    HawkesMarkCallback sampler, void* sampler_ctx,
                                    double** times_out, double** marks_out) {
    if (!times_out || !marks_out) {
        return handle_failure(times_out, marks_out, "null output pointer");
    }
    if (!(mu > 0.0) || !(horizon >= 0.0)) {
        return handle_failure(times_out, marks_out, "invalid Poisson parameters");
    }

    g_last_error.clear();
    order_flow::PoissonProcess process(mu);
    CallbackSampler cb{sampler, sampler_ctx};
    order_flow::PoissonProcess::MarkSampler mark_sampler =
        [cb](std::mt19937_64& rng) mutable { return cb(rng); };
    auto result = process.simulate(horizon, mark_sampler, seed);
    return copy_stream(result, times_out, marks_out);
}

void hawkes_free(double* ptr) {
    std::free(ptr);
}

const char* hawkes_last_error() {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

}  // extern "C"
