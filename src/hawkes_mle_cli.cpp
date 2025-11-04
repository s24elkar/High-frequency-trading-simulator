#include "order_flow/HawkesMLE.hpp"
#include "order_flow/OrderFlow.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "error.hpp"

namespace {

using order_flow::EventStream;
using order_flow::ExponentialKernel;
using order_flow::HawkesProcess;
using order_flow::calibration::HawkesMLEConfig;
using order_flow::calibration::HawkesMLEResult;
using order_flow::calibration::HawkesParameters;

struct Options {
    double mu{0.6};
    double alpha{0.4};
    double beta{1.2};
    double horizon{200.0};
    std::size_t replicates{100};
    std::uint64_t seed{1337};
    std::string output_dir{"results/week5/calibration"};
    bool store_events{true};
};

Options parse_arguments(int argc, char** argv) {
    Options opts;
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key(argv[i]);
        std::string value(argv[i + 1]);
        args[key] = value;
    }

    auto maybe_to_double = [](const std::string& value, double fallback) {
        try {
            return std::stod(value);
        } catch (...) {
            return fallback;
        }
    };

    auto maybe_to_size = [](const std::string& value, std::size_t fallback) {
        try {
            return static_cast<std::size_t>(std::stoull(value));
        } catch (...) {
            return fallback;
        }
    };

    auto maybe_to_seed = [](const std::string& value, std::uint64_t fallback) {
        try {
            return static_cast<std::uint64_t>(std::stoull(value));
        } catch (...) {
            return fallback;
        }
    };

    if (args.count("--mu")) {
        opts.mu = maybe_to_double(args["--mu"], opts.mu);
    }
    if (args.count("--alpha")) {
        opts.alpha = maybe_to_double(args["--alpha"], opts.alpha);
    }
    if (args.count("--beta")) {
        opts.beta = maybe_to_double(args["--beta"], opts.beta);
    }
    if (args.count("--horizon")) {
        opts.horizon = maybe_to_double(args["--horizon"], opts.horizon);
    }
    if (args.count("--replicates")) {
        opts.replicates = maybe_to_size(args["--replicates"], opts.replicates);
    }
    if (args.count("--seed")) {
        opts.seed = maybe_to_seed(args["--seed"], opts.seed);
    }
    if (args.count("--output")) {
        opts.output_dir = args["--output"];
    }
    if (args.count("--store-events")) {
        const std::string flag = args["--store-events"];
        opts.store_events = (flag == "1" || flag == "true" || flag == "yes");
    }
    return opts;
}

void ensure_directory(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        std::ostringstream oss;
        oss << "Failed to create directory " << path << ": " << ec.message();
        HFT_THROW(std::runtime_error(oss.str()));
    }
}

struct ReplicateSummary {
    std::size_t index;
    std::uint64_t seed;
    std::size_t events;
    bool converged;
    double mu_hat;
    double alpha_hat;
    double beta_hat;
    double log_likelihood;
    double gradient_norm;
    double branching_ratio;
    std::size_t iterations;
};

void write_summary_csv(const std::filesystem::path& path, const std::vector<ReplicateSummary>& summaries) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Failed to open summary CSV for writing"));
    }
    out << "replicate,seed,event_count,converged,mu_hat,alpha_hat,beta_hat,log_likelihood,gradient_norm,branching_ratio,iterations\n";
    out << std::setprecision(10);
    for (const auto& row : summaries) {
        out << row.index << ','
            << row.seed << ','
            << row.events << ','
            << (row.converged ? 1 : 0) << ','
            << row.mu_hat << ','
            << row.alpha_hat << ','
            << row.beta_hat << ','
            << row.log_likelihood << ','
            << row.gradient_norm << ','
            << row.branching_ratio << ','
            << row.iterations << '\n';
    }
}

void write_trajectory_csv(const std::filesystem::path& path, const std::vector<HawkesParameters>& trajectory) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Failed to open trajectory CSV"));
    }
    out << "iteration,mu,alpha,beta\n";
    out << std::setprecision(10);
    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        out << i << ',' << trajectory[i].mu << ',' << trajectory[i].alpha << ',' << trajectory[i].beta << '\n';
    }
}

void write_events_csv(const std::filesystem::path& path, const EventStream& events) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Failed to open events CSV"));
    }
    out << "time,mark\n";
    out << std::setprecision(10);
    for (const auto& event : events.data()) {
        out << event.time << ',' << event.mark << '\n';
    }
}

HawkesMLEConfig default_config() {
    HawkesMLEConfig config;
    config.gradient_tolerance = 5e-5;
    config.parameter_tolerance = 1e-5;
    config.max_iterations = 400;
    config.max_branching_ratio = 0.999;
    config.enforce_stationarity = true;
    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_arguments(argc, argv);
        const std::filesystem::path output_dir(opts.output_dir);
        const auto trajectories_dir = output_dir / "trajectories";
        const auto events_dir = output_dir / "events";

        ensure_directory(output_dir);
        ensure_directory(trajectories_dir);
        if (opts.store_events) {
            ensure_directory(events_dir);
        }

        std::vector<ReplicateSummary> summaries;
        summaries.reserve(opts.replicates);

        auto kernel = std::make_shared<ExponentialKernel>(opts.alpha, opts.beta);
        HawkesProcess process(opts.mu, kernel);
        process.set_mark_expectation(1.0);
        HawkesMLEConfig config = default_config();

        for (std::size_t i = 0; i < opts.replicates; ++i) {
            const std::uint64_t replicate_seed = opts.seed + static_cast<std::uint64_t>(i);
            const EventStream events = process.simulate(opts.horizon, replicate_seed);

            auto result = order_flow::calibration::fit_exponential_hawkes_mle(events, opts.horizon, config);

            if (!result.converged) {
                HawkesMLEConfig retry = config;
                retry.max_iterations = 600;
                retry.gradient_tolerance = 1e-4;
                retry.parameter_tolerance = 1e-4;
                retry.enforce_stationarity = false;
                retry.initial_guess = result.params;
                auto retry_result = order_flow::calibration::fit_exponential_hawkes_mle(events, opts.horizon, retry);
                if (retry_result.converged) {
                    result = std::move(retry_result);
                } else {
                    HawkesMLEConfig alternate = config;
                    alternate.max_iterations = 800;
                    alternate.gradient_tolerance = 5e-4;
                    alternate.parameter_tolerance = 5e-4;
                    alternate.enforce_stationarity = false;
                    const double mu0 = std::max(1e-3, static_cast<double>(events.size()) / std::max(opts.horizon, 1.0));
                    const double alpha0 = std::max(1e-3, 0.5 * mu0);
                    const double beta0 = 1.0;
                    alternate.initial_guess = HawkesParameters{mu0, alpha0, beta0};
                    auto alternate_result = order_flow::calibration::fit_exponential_hawkes_mle(events, opts.horizon, alternate);
                    if (alternate_result.converged) {
                        result = std::move(alternate_result);
                    }
                }
            }

            const double branching = order_flow::calibration::branching_ratio(result.params);

            summaries.push_back(ReplicateSummary{
                i,
                replicate_seed,
                events.size(),
                result.converged,
                result.params.mu,
                result.params.alpha,
                result.params.beta,
                result.log_likelihood,
                result.gradient_norm,
                branching,
                result.iterations});

            const auto trajectory_path = trajectories_dir / ("trajectory_" + std::to_string(i) + ".csv");
            write_trajectory_csv(trajectory_path, result.trajectory);

            if (opts.store_events && i == 0) {
                const auto events_path = events_dir / "replicate_0.csv";
                write_events_csv(events_path, events);
            }
        }

        const auto summary_path = output_dir / "replicate_summary.csv";
        write_summary_csv(summary_path, summaries);

        std::cout << "Saved calibration summaries to " << summary_path << std::endl;
        std::cout << "Trajectories stored in " << trajectories_dir << std::endl;
        if (opts.store_events) {
            std::cout << "Events stored in " << events_dir << std::endl;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "hawkes_mle_cli error: " << ex.what() << std::endl;
        return 1;
    }
}
