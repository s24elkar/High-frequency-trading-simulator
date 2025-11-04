#include "order_flow/HawkesMLE.hpp"
#include "order_flow/OrderFlow.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <memory>
#include <tuple>
#include <vector>

#include "error.hpp"

namespace {

using order_flow::EventStream;
using order_flow::ExponentialKernel;
using order_flow::HawkesProcess;
using order_flow::calibration::HawkesMLEConfig;
using order_flow::calibration::HawkesMLEResult;
using order_flow::calibration::HawkesParameters;

struct Scenario {
    std::string label;
    double mu;
    double alpha;
    double beta;
    double horizon;
    std::size_t replicates;
    std::uint64_t seed;
    double dropout_rate;
};

struct Options {
    std::string output_dir{"results/week5/robustness/native_validation"};
    std::string scenario_filter{};
    std::size_t replicates_override{0};
    double dropout_override{-1.0};
};

struct ReplicateRow {
    std::size_t replicate{};
    std::uint64_t seed{};
    std::size_t simulated_events{};
    std::size_t retained_events{};
    bool converged{};
    double mu_hat{};
    double alpha_hat{};
    double beta_hat{};
    double log_likelihood{};
    double gradient_norm{};
    double branching_ratio{};
    std::size_t iterations{};
};

std::vector<Scenario> default_scenarios() {
    return {
        {"BTCUSDT", 0.52, 0.36, 1.55, 3600.0, 200, 1337, 0.01},
        {"ETHUSDT", 0.44, 0.30, 1.40, 3600.0, 200, 2337, 0.015},
        {"BNBUSDT", 0.38, 0.27, 1.25, 3600.0, 200, 3337, 0.02},
        {"SOLUSDT", 0.31, 0.22, 1.15, 3600.0, 200, 4337, 0.02},
    };
}

Options parse_arguments(int argc, char** argv) {
    Options opts;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key(argv[i]);
        const std::string value(argv[i + 1]);
        if (key == "--output") {
            opts.output_dir = value;
        } else if (key == "--scenario") {
            opts.scenario_filter = value;
        } else if (key == "--replicates") {
            opts.replicates_override = static_cast<std::size_t>(std::stoull(value));
        } else if (key == "--dropout") {
            opts.dropout_override = std::stod(value);
        } else {
            std::cerr << "Unknown argument: " << key << '\n';
        }
    }
    return opts;
}

void ensure_directory(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        HFT_THROW(std::runtime_error("Failed to create directory: " + path.string()));
    }
}

std::tuple<EventStream, std::size_t> apply_dropout(
    const EventStream& input,
    double rate,
    std::uint64_t seed) {
    if (rate <= 0.0) {
        return {input, 0};
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    EventStream filtered;
    filtered.reserve(input.size());
    std::size_t dropped = 0;
    for (const auto& event : input.data()) {
        if (dist(rng) < rate) {
            ++dropped;
            continue;
        }
        filtered.add(event.time, event.mark);
    }
    return {filtered, dropped};
}

void write_summary_csv(
    const std::filesystem::path& path,
    const std::vector<ReplicateRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Unable to open summary CSV: " + path.string()));
    }
    out << "replicate,seed,simulated_events,retained_events,converged,mu_hat,alpha_hat,beta_hat,log_likelihood,gradient_norm,branching_ratio,iterations\n";
    out << std::setprecision(10);
    for (const auto& row : rows) {
        out << row.replicate << ','
            << row.seed << ','
            << row.simulated_events << ','
            << row.retained_events << ','
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

void write_metadata(
    const std::filesystem::path& path,
    const Scenario& scenario,
    std::size_t replicates,
    double dropout_rate) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Unable to write metadata: " + path.string()));
    }
    out << "{\n"
        << "  \"label\": \"" << scenario.label << "\",\n"
        << "  \"mu\": " << scenario.mu << ",\n"
        << "  \"alpha\": " << scenario.alpha << ",\n"
        << "  \"beta\": " << scenario.beta << ",\n"
        << "  \"rho\": " << (scenario.beta > 0.0 ? scenario.alpha / scenario.beta : 0.0) << ",\n"
        << "  \"horizon\": " << scenario.horizon << ",\n"
        << "  \"replicates\": " << replicates << ",\n"
        << "  \"seed\": " << scenario.seed << ",\n"
        << "  \"dropout_rate\": " << dropout_rate << "\n"
        << "}\n";
}

ReplicateRow evaluate_replicate(
    const Scenario& scenario,
    std::size_t replicate,
    std::uint64_t seed,
    double dropout_rate,
    const HawkesMLEConfig& config) {
    auto kernel = std::make_shared<ExponentialKernel>(scenario.alpha, scenario.beta);
    HawkesProcess process(scenario.mu, kernel);
    process.set_mark_expectation(1.0);

    const EventStream simulated = process.simulate(scenario.horizon, seed);
    auto [observed, dropped] = apply_dropout(simulated, dropout_rate, seed + 1024);

    const HawkesMLEResult fit =
        order_flow::calibration::fit_exponential_hawkes_mle(observed, scenario.horizon, config);
    const HawkesParameters& params = fit.params;

    ReplicateRow row;
    row.replicate = replicate;
    row.seed = seed;
    row.simulated_events = simulated.size();
    row.retained_events = observed.size();
    row.converged = fit.converged;
    row.mu_hat = params.mu;
    row.alpha_hat = params.alpha;
    row.beta_hat = params.beta;
    row.log_likelihood = fit.log_likelihood;
    row.gradient_norm = fit.gradient_norm;
    row.branching_ratio = (params.beta > 0.0) ? params.alpha / params.beta : 0.0;
    row.iterations = fit.iterations;
    return row;
}

void run_scenario(
    const Scenario& scenario,
    const Options& options,
    const std::filesystem::path& root) {
    const std::size_t replicates =
        options.replicates_override > 0 ? options.replicates_override : scenario.replicates;
    const double dropout =
        options.dropout_override >= 0.0 ? options.dropout_override : scenario.dropout_rate;

    const auto scenario_dir = root / scenario.label;
    ensure_directory(scenario_dir);

    HawkesMLEConfig config;
    config.gradient_tolerance = 5e-5;
    config.parameter_tolerance = 1e-5;
    config.max_iterations = 400;
    config.enforce_stationarity = true;
    config.max_branching_ratio = 0.999;

    std::vector<ReplicateRow> rows;
    rows.reserve(replicates);

    for (std::size_t i = 0; i < replicates; ++i) {
        const std::uint64_t seed = scenario.seed + static_cast<std::uint64_t>(i);
        rows.push_back(evaluate_replicate(scenario, i, seed, dropout, config));
    }

    write_summary_csv(scenario_dir / "replicates.csv", rows);
    write_metadata(scenario_dir / "metadata.json", scenario, replicates, dropout);
    std::cout << "Scenario " << scenario.label << ": wrote " << rows.size()
              << " rows to " << (scenario_dir / "replicates.csv") << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parse_arguments(argc, argv);
    const std::filesystem::path output_root(options.output_dir);
    ensure_directory(output_root);

    std::vector<Scenario> scenarios = default_scenarios();
    if (!options.scenario_filter.empty()) {
        std::vector<Scenario> filtered;
        for (const auto& scenario : scenarios) {
            if (scenario.label == options.scenario_filter) {
                filtered.push_back(scenario);
            }
        }
        if (filtered.empty()) {
            std::cerr << "No scenario matched filter: " << options.scenario_filter << '\n';
            return 1;
        }
        scenarios = std::move(filtered);
    }

    for (const auto& scenario : scenarios) {
        run_scenario(scenario, options, output_root);
    }
    return 0;
}
