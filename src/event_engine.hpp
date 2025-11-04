#pragma once

#include "hawkes_engine.hpp"
#include "thread_utils.hpp"
#include "memory_pool.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <boost/lockfree/spsc_queue.hpp>

namespace simulator {

struct SimulationConfig;

struct EventEnvelope {
    HawkesEvent event{};
    IntensityBuffer intensities;
    std::chrono::high_resolution_clock::time_point produced_at{};
    bool is_termination{false};
};

using EventQueue = boost::lockfree::spsc_queue<EventEnvelope>;

class EventEngine {
public:
    EventEngine(const SimulationConfig& config,
                EventQueue& queue,
                std::atomic<bool>& shutdown_signal,
                std::atomic<bool>& finished_flag,
                concurrency::ThreadConfig thread_config);
    ~EventEngine();

    void start();
    void join();

    [[nodiscard]] concurrency::ThreadStats stats() const noexcept;
    [[nodiscard]] std::size_t produced_events() const noexcept;
    [[nodiscard]] std::size_t queue_retries() const noexcept;

private:
    void run();
    bool should_stop(double event_time, std::size_t produced) const;

    const SimulationConfig& config_;
    EventQueue& queue_;
    std::atomic<bool>& shutdown_signal_;
    std::atomic<bool>& finished_flag_;
    concurrency::ThreadConfig thread_config_;
    std::thread thread_;
    concurrency::ThreadStats stats_{};
    std::atomic<std::size_t> produced_{0};
    std::atomic<std::size_t> push_retries_{0};
};

} // namespace simulator
