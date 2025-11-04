#include "engine_runtime.hpp"
#include "thread_utils.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr std::size_t kDefaultQueueCapacity = 1 << 18;
constexpr std::size_t kDefaultEvents = 10'000'000;
constexpr double kDefaultMinutes = 60.0;

struct StressOptions {
    std::size_t events{kDefaultEvents};
    double minutes{kDefaultMinutes};
    double intensity_scale{1.0};
    std::filesystem::path metrics_path{"results/week6/stress_test_metrics.csv"};
};

StressOptions parse_args(int argc, char** argv) {
    StressOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if ((arg == "--events" || arg == "-e") && i + 1 < argc) {
            opts.events = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if ((arg == "--minutes" || arg == "-m") && i + 1 < argc) {
            opts.minutes = std::stod(argv[++i]);
        } else if ((arg == "--scale" || arg == "-s") && i + 1 < argc) {
            opts.intensity_scale = std::stod(argv[++i]);
        } else if ((arg == "--metrics" || arg == "-o") && i + 1 < argc) {
            opts.metrics_path = argv[++i];
        }
    }
    if (opts.minutes <= 0.0) {
        opts.minutes = kDefaultMinutes;
    }
    if (opts.events == 0) {
        opts.events = kDefaultEvents;
    }
    if (!(opts.intensity_scale > 0.0)) {
        opts.intensity_scale = 1.0;
    }
    return opts;
}

void ensure_parent_dir(const std::filesystem::path& path) {
    if (const auto parent = path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_metrics_csv(const simulator::EngineRuntimeMetrics& metrics,
                       const std::filesystem::path& output_path) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Unable to write metrics CSV: " << output_path << '\n';
        return;
    }
    out << std::setprecision(6) << std::fixed;
    out << "metric,value\n";
    out << "events_generated," << metrics.events_generated << '\n';
    out << "events_processed," << metrics.events_processed << '\n';
    out << "event_queue_retries," << metrics.event_queue_retries << '\n';
    out << "match_queue_retries," << metrics.match_queue_retries << '\n';
    out << "mean_latency_us," << metrics.logger_metrics.mean_inter_thread_latency_us << '\n';
    out << "p99_latency_us," << metrics.logger_metrics.p99_inter_thread_latency_us << '\n';
    out << "max_latency_us," << metrics.logger_metrics.max_inter_thread_latency_us << '\n';
    out << "mean_arrival_latency_us," << metrics.logger_metrics.mean_arrival_latency_us << '\n';
    out << "throughput_eps," << metrics.logger_metrics.throughput_eps << '\n';
    out << "event_cpu_util," << metrics.event_thread.utilization << '\n';
    out << "match_cpu_util," << metrics.match_thread.utilization << '\n';
    out << "log_cpu_util," << metrics.logger_thread.utilization << '\n';

    out << "\ntype,timestamp_s,cpu_seconds,cpu_utilization,rss_mb\n";
    for (const auto& sample : metrics.resource_samples) {
        out << "resource," << sample.timestamp_s << ','
            << sample.cpu_seconds << ','
            << sample.cpu_utilization << ','
            << sample.rss_mb << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    const StressOptions opts = parse_args(argc, argv);

    simulator::SimulationConfig config = simulator::default_btcusdt_config();
    config.session_length = opts.minutes * 60.0;
    config.max_events = opts.events;
    config.event_log_path.clear();
    if (opts.intensity_scale != 1.0) {
        for (double& mu_value : config.mu) {
            mu_value *= opts.intensity_scale;
        }
        for (auto& row : config.alpha) {
            for (double& value : row) {
                value *= opts.intensity_scale;
            }
        }
    }

    simulator::concurrency::ThreadConfig event_thread{};
    event_thread.name = "stress-event";
    simulator::concurrency::ThreadConfig match_thread{};
    match_thread.name = "stress-match";
    simulator::concurrency::ThreadConfig log_thread{};
    log_thread.name = "stress-log";

    simulator::EngineRuntime runtime(config,
                                     kDefaultQueueCapacity,
                                     event_thread,
                                     match_thread,
                                     log_thread);

    std::cout << "Starting stress test: " << opts.events << " events, "
              << opts.minutes << " minutes horizon, scale "
              << opts.intensity_scale << std::endl;
    const auto metrics = runtime.run();
    write_metrics_csv(metrics, opts.metrics_path);

    std::cout << "Generated events : " << metrics.events_generated << '\n'
              << "Processed events : " << metrics.events_processed << '\n'
              << "Mean latency us  : " << metrics.logger_metrics.mean_inter_thread_latency_us << '\n'
              << "p99 latency us   : " << metrics.logger_metrics.p99_inter_thread_latency_us << '\n'
              << "Max latency us   : " << metrics.logger_metrics.max_inter_thread_latency_us << '\n'
              << "Event queue retries: " << metrics.event_queue_retries << '\n'
              << "Match queue retries: " << metrics.match_queue_retries << '\n'
              << "Resource samples recorded: " << metrics.resource_samples.size() << std::endl;

    return 0;
}
