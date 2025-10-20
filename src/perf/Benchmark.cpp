#include "simulator_core.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/time.h>

namespace {

struct BenchmarkConfig {
    std::size_t events{1'000'000};
    double session_length{1'000'000.0};
    double intensity_scale{12.0};
    std::filesystem::path output_csv{"results/week6/performance_benchmark.csv"};
    std::string label{"optimized"};
};

std::optional<BenchmarkConfig> parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--events" && i + 1 < argc) {
            cfg.events = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--session" && i + 1 < argc) {
            cfg.session_length = std::stod(argv[++i]);
        } else if (arg == "--scale" && i + 1 < argc) {
            cfg.intensity_scale = std::stod(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_csv = argv[++i];
        } else if (arg == "--label" && i + 1 < argc) {
            cfg.label = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: perf_benchmark [--events N] [--session T] [--scale S] [--output PATH] [--label NAME]\n";
            return std::nullopt;
        }
    }
    if (cfg.events == 0) {
        cfg.events = 1'000'000;
    }
    if (cfg.session_length <= 0.0) {
        cfg.session_length = 1'000'000.0;
    }
    if (cfg.intensity_scale <= 0.0) {
        cfg.intensity_scale = 12.0;
    }
    return cfg;
}

double compute_mean_latency_us(const simulator::SimulationResult& result) {
    if (result.arrivals.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& arrival : result.arrivals) {
        total += arrival.latency;
    }
    const double mean_seconds = total / static_cast<double>(result.arrivals.size());
    return mean_seconds * 1'000'000.0;
}

double resident_memory_mb() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    constexpr double kBytesToMB = 1.0 / (1024.0 * 1024.0);
    return static_cast<double>(usage.ru_maxrss) * kBytesToMB;
#else
    // ru_maxrss is expressed in kilobytes on Linux
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

void write_csv_row(const BenchmarkConfig& cfg,
                   std::size_t events,
                   double elapsed_seconds,
                   double throughput,
                   double latency_us,
                   double memory_mb) {
    std::filesystem::create_directories(cfg.output_csv.parent_path());
    const bool exists = std::filesystem::exists(cfg.output_csv);
    std::ofstream out(cfg.output_csv, std::ios::app);
    if (!exists) {
        out << "label,events,elapsed_seconds,throughput_eps,mean_latency_us,max_rss_mb\n";
    }
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(6);
    out << cfg.label << ','
        << events << ','
        << elapsed_seconds << ','
        << throughput << ','
        << latency_us << ','
        << memory_mb << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const auto parsed = parse_args(argc, argv);
    if (!parsed) {
        return 0;
    }
    const BenchmarkConfig cfg = *parsed;

    simulator::SimulationConfig sim_cfg = simulator::default_btcusdt_config();
    for (double& mu_value : sim_cfg.mu) {
        mu_value *= cfg.intensity_scale;
    }
    for (auto& row : sim_cfg.alpha) {
        for (double& value : row) {
            value *= cfg.intensity_scale;
        }
    }
    sim_cfg.session_length = cfg.session_length;
    sim_cfg.max_events = cfg.events;
    sim_cfg.event_log_path.clear();

    simulator::SimulatorCore core(sim_cfg);

    const auto start = std::chrono::steady_clock::now();
    const simulator::SimulationResult result = core.run();
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
    const std::size_t processed = result.arrivals.size();
    const double throughput = processed > 0 && elapsed_seconds > 0.0
                                  ? static_cast<double>(processed) / elapsed_seconds
                                  : 0.0;
    const double latency_us = compute_mean_latency_us(result);
    const double memory_mb = resident_memory_mb();

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout << "Benchmark label: " << cfg.label << '\n'
              << "Events processed: " << processed << '\n'
              << "Elapsed seconds : " << elapsed_seconds << '\n'
              << "Throughput (eps): " << throughput << '\n'
              << "Mean latency (us): " << latency_us << '\n'
              << "Max RSS (MB)     : " << memory_mb << '\n';

    write_csv_row(cfg, processed, elapsed_seconds, throughput, latency_us, memory_mb);

    return processed >= cfg.events ? EXIT_SUCCESS : EXIT_FAILURE;
}

