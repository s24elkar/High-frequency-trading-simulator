#include "event_engine.hpp"

#include "simulator_core.hpp"

#include <random>
#include <thread>
#include <utility>

namespace simulator {

EventEngine::EventEngine(const SimulationConfig& config,
                         EventQueue& queue,
                         std::atomic<bool>& shutdown_signal,
                         std::atomic<bool>& finished_flag,
                         concurrency::ThreadConfig thread_config)
    : config_(config),
      queue_(queue),
      shutdown_signal_(shutdown_signal),
      finished_flag_(finished_flag),
      thread_config_(std::move(thread_config)) {
    if (thread_config_.name.empty()) {
        thread_config_.name = "event-engine";
    }
}

EventEngine::~EventEngine() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void EventEngine::start() {
    thread_ = std::thread(&EventEngine::run, this);
}

void EventEngine::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

concurrency::ThreadStats EventEngine::stats() const noexcept {
    return stats_;
}

std::size_t EventEngine::produced_events() const noexcept {
    return produced_.load(std::memory_order_relaxed);
}

std::size_t EventEngine::queue_retries() const noexcept {
    return push_retries_.load(std::memory_order_relaxed);
}

bool EventEngine::should_stop(double event_time, std::size_t produced) const {
    if (config_.max_events > 0 && produced >= config_.max_events) {
        return true;
    }
    return event_time >= config_.session_length;
}

void EventEngine::run() {
    concurrency::apply_thread_config(thread_config_);
    const auto timing = concurrency::begin_timing();

    std::mt19937_64 rng(config_.seed);
    ExponentialHawkesProcess process(config_.mu, config_.alpha, config_.beta);
    process.reset(0.0);

    while (!shutdown_signal_.load(std::memory_order_relaxed)) {
        HawkesEvent next = process.sample_next(rng);
        EventEnvelope envelope;
        envelope.event = next;
        const auto& lambda_snapshot = process.intensities();
        envelope.intensities.assign(lambda_snapshot.begin(), lambda_snapshot.end());
        envelope.produced_at = std::chrono::high_resolution_clock::now();
        envelope.is_termination = false;

        while (!queue_.push(envelope)) {
            if (shutdown_signal_.load(std::memory_order_relaxed)) {
                break;
            }
            push_retries_.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }

        const std::size_t produced = produced_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (should_stop(next.time, produced)) {
            break;
        }
    }

    EventEnvelope sentinel;
    sentinel.is_termination = true;
    sentinel.produced_at = std::chrono::high_resolution_clock::now();
    while (!queue_.push(sentinel)) {
        if (shutdown_signal_.load(std::memory_order_relaxed)) {
            break;
        }
        push_retries_.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    }

    stats_ = concurrency::end_timing(timing);
    finished_flag_.store(true, std::memory_order_release);
}

} // namespace simulator
