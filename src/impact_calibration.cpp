#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "error.hpp"

#include <zlib.h>
#include <random>

namespace {

struct Options {
    std::filesystem::path input_path;
    std::filesystem::path output_dir{"results/week6/impact_calibration"};
    std::string symbol_override{};
    std::size_t lookback_trades{5};
    std::size_t lookahead_trades{5};
    std::int64_t depth_window_ms{60'000};
    double min_abs_qty{1e-5};
    double min_depth{1e-3};
    double min_positive_residual{1e-8};
    double min_relative_size{1e-3};
    std::size_t max_samples{0};
    std::uint64_t sample_seed{1337};
};

struct TradeRecord {
    std::int64_t ts_ms{};
    double price{};
    double qty{};
    int direction{0}; // +1 for aggressive buy, -1 for aggressive sell
};

struct Sample {
    double volume{};
    double slippage{};
    double reference_price{};
    double depth{};
};

struct Metrics {
    double rmse{};
    double mae{};
    double r2{};
    double mse{};
};

class LineReader {
public:
    explicit LineReader(const std::filesystem::path& path) {
        const std::string ext = path.extension().string();
        const bool is_gzip = (ext == ".gz");
        if (is_gzip) {
            mode_ = Mode::Gzip;
            gz_ = gzopen(path.string().c_str(), "rb");
            if (!gz_) {
                HFT_THROW(std::runtime_error("Failed to open gzip file: " + path.string()));
            }
            gzbuffer(gz_, 1 << 20); // 1 MiB internal buffer to reduce syscall churn
        } else {
            mode_ = Mode::Plain;
            plain_.open(path);
            if (!plain_) {
                HFT_THROW(std::runtime_error("Failed to open file: " + path.string()));
            }
        }
    }

    ~LineReader() {
        if (mode_ == Mode::Gzip && gz_) {
            gzclose(gz_);
        }
    }

    bool getline(std::string& out) {
        out.clear();
        if (mode_ == Mode::Plain) {
            return static_cast<bool>(std::getline(plain_, out));
        }

        if (!gz_) {
            return false;
        }

        std::string buffer;
        buffer.reserve(512);
        char chunk[4096];
        while (true) {
            char* res = gzgets(gz_, chunk, static_cast<int>(sizeof(chunk)));
            if (!res) {
                if (buffer.empty()) {
                    return false;
                }
                break;
            }
            buffer.append(res);
            if (!buffer.empty() && buffer.back() == '\n') {
                break;
            }
            if (gzeof(gz_) != 0) {
                break;
            }
        }

        if (buffer.empty()) {
            return false;
        }
        while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r')) {
            buffer.pop_back();
        }
        out = std::move(buffer);
        return true;
    }

private:
    enum class Mode { Plain, Gzip };
    Mode mode_{Mode::Plain};
    std::ifstream plain_;
    gzFile gz_{nullptr};
};

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    fields.reserve(16);
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

Options parse_arguments(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input" && i + 1 < argc) {
            opts.input_path = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            opts.output_dir = argv[++i];
        } else if (arg == "--symbol" && i + 1 < argc) {
            opts.symbol_override = argv[++i];
        } else if (arg == "--lookback-trades" && i + 1 < argc) {
            opts.lookback_trades = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--lookahead-trades" && i + 1 < argc) {
            opts.lookahead_trades = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--depth-window-ms" && i + 1 < argc) {
            opts.depth_window_ms = static_cast<std::int64_t>(std::stoll(argv[++i]));
        } else if (arg == "--min-abs-qty" && i + 1 < argc) {
            opts.min_abs_qty = std::stod(argv[++i]);
        } else if (arg == "--min-depth" && i + 1 < argc) {
            opts.min_depth = std::stod(argv[++i]);
        } else if (arg == "--min-residual" && i + 1 < argc) {
            opts.min_positive_residual = std::stod(argv[++i]);
        } else if (arg == "--min-relative-size" && i + 1 < argc) {
            opts.min_relative_size = std::stod(argv[++i]);
        } else if (arg == "--max-samples" && i + 1 < argc) {
            opts.max_samples = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--sample-seed" && i + 1 < argc) {
            opts.sample_seed = std::stoull(argv[++i]);
        } else if (arg == "--help") {
            std::cout
                << "Usage: impact_calibration --input PATH [options]\n"
                << "Options:\n"
                << "  --output-dir PATH           Directory for calibration artifacts\n"
                << "  --symbol SYMBOL             Override symbol label in output\n"
                << "  --lookback-trades N         Trades to average for pre-trade price (default: 5)\n"
                << "  --lookahead-trades N        Trades to average for post-trade price (default: 5)\n"
                << "  --depth-window-ms MS        Rolling window for depth estimation (default: 60000)\n"
                << "  --min-abs-qty Q             Minimum absolute trade size to include (default: 1e-5)\n"
                << "  --min-depth Q               Minimum rolling depth for liquidity term (default: 1e-3)\n"
                << "  --min-residual VALUE        Floor for residuals in liquidity regression (default: 1e-8)\n"
                << "  --min-relative-size VALUE   Minimum v/Q ratio considered (default: 1e-3)\n"
                << "  --max-samples N             Optional cap on samples via reservoir sampling\n"
                << "  --sample-seed SEED          Seed for reservoir sampling (default: 1337)\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
        }
    }
    return opts;
}

std::string infer_symbol(const std::filesystem::path& path) {
    const std::string stem = path.stem().string();
    const std::string marker = "-trades-";
    if (auto pos = stem.find(marker); pos != std::string::npos) {
        return stem.substr(0, pos);
    }
    return stem;
}

std::vector<TradeRecord> load_trades(const Options& opts, std::string& symbol_out) {
    if (opts.input_path.empty()) {
        HFT_THROW(std::runtime_error("Missing --input argument"));
    }
    if (!std::filesystem::exists(opts.input_path)) {
        HFT_THROW(std::runtime_error("Input file not found: " + opts.input_path.string()));
    }

    symbol_out = !opts.symbol_override.empty() ? opts.symbol_override : infer_symbol(opts.input_path);

    LineReader reader(opts.input_path);
    std::string line;
    if (!reader.getline(line)) {
        HFT_THROW(std::runtime_error("Input file is empty: " + opts.input_path.string()));
    }

    std::vector<TradeRecord> trades;
    trades.reserve(1'000'000);
    std::size_t malformed = 0;

    auto parse_int64 = [](const std::string& input, std::int64_t& out) -> bool {
        char* end = nullptr;
        errno = 0;
        const long long value = std::strtoll(input.c_str(), &end, 10);
        if (end == input.c_str() || (end && *end != '\0') || errno == ERANGE) {
            return false;
        }
        out = static_cast<std::int64_t>(value);
        return true;
    };

    auto parse_double = [](const std::string& input, double& out) -> bool {
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(input.c_str(), &end);
        if (end == input.c_str() || (end && *end != '\0') || errno == ERANGE || !std::isfinite(value)) {
            return false;
        }
        out = value;
        return true;
    };

    while (reader.getline(line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_csv_row(line);
        if (fields.size() < 8) {
            ++malformed;
            continue;
        }
        TradeRecord record;
        if (!parse_int64(fields[2], record.ts_ms) ||
            !parse_double(fields[4], record.price) ||
            !parse_double(fields[5], record.qty)) {
            ++malformed;
            continue;
        }
        const std::string& side = fields[7];
        record.direction = (side == "buy") ? 1 : (side == "sell" ? -1 : 0);
        if (record.direction == 0) {
            ++malformed;
            continue;
        }
        trades.push_back(record);
    }

    if (!malformed) {
        std::cerr << "Loaded " << trades.size() << " trades for " << symbol_out << '\n';
    } else {
        std::cerr << "Loaded " << trades.size() << " trades for " << symbol_out
                  << " (" << malformed << " malformed row(s) skipped)\n";
    }
    return trades;
}

std::vector<double> compute_depths(
    const std::vector<TradeRecord>& trades,
    std::int64_t window_ms
) {
    std::vector<double> depths(trades.size(), 0.0);
    std::deque<std::pair<std::int64_t, double>> window;
    double running_sum = 0.0;

    for (std::size_t i = 0; i < trades.size(); ++i) {
        const auto& tr = trades[i];
        while (!window.empty() && (tr.ts_ms - window.front().first) > window_ms) {
            running_sum -= window.front().second;
            window.pop_front();
        }
        depths[i] = running_sum;
        const double volume = std::fabs(tr.qty);
        running_sum += volume;
        window.emplace_back(tr.ts_ms, volume);
    }
    return depths;
}

std::vector<Sample> build_samples(
    const std::vector<TradeRecord>& trades,
    const std::vector<double>& depths,
    const Options& opts
) {
    const std::size_t n = trades.size();
    if (n == 0) {
        return {};
    }

    if (opts.lookback_trades == 0) {
        HFT_THROW(std::runtime_error("lookback-trades must be positive"));
    }

    std::vector<double> prefix_price(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        prefix_price[i + 1] = prefix_price[i] + trades[i].price;
    }

    std::vector<Sample> samples;
    samples.reserve(n / 4); // heuristic

    std::size_t start_index = opts.lookback_trades;
    std::size_t end_index = n;

    if (start_index >= end_index) {
        return samples;
    }

    std::size_t seen = 0;
    std::vector<Sample> reservoir;
    std::optional<std::mt19937_64> rng;
    if (opts.max_samples != 0) {
        reservoir.reserve(opts.max_samples);
        rng.emplace(opts.sample_seed);
    }

    for (std::size_t i = start_index; i < end_index; ++i) {
        const TradeRecord& tr = trades[i];
        const double volume = std::fabs(tr.qty);
        if (volume < opts.min_abs_qty) {
            continue;
        }

        const std::size_t prev_start = i - opts.lookback_trades;
        const std::size_t prev_end = i;
        const double prev_sum = prefix_price[prev_end] - prefix_price[prev_start];
        double reference_price = prev_sum / static_cast<double>(opts.lookback_trades);

        if (!std::isfinite(reference_price) || reference_price <= 0.0) {
            continue;
        }

        const double execution_price = tr.price;
        const double raw_slippage = tr.direction * ((execution_price - reference_price) / reference_price);
        const double slippage = std::fabs(raw_slippage);
        if (!std::isfinite(slippage)) {
            continue;
        }

        Sample sample;
        sample.volume = volume;
        sample.slippage = slippage;
        sample.reference_price = reference_price;
        sample.depth = depths[i];

        if (opts.max_samples == 0) {
            samples.push_back(sample);
        } else {
            ++seen;
            if (reservoir.size() < opts.max_samples) {
                reservoir.push_back(sample);
            } else {
                std::uniform_int_distribution<std::size_t> dist(0, seen - 1);
                const std::size_t idx = dist(*rng);
                if (idx < opts.max_samples) {
                    reservoir[idx] = sample;
                }
            }
        }
    }

    if (opts.max_samples != 0) {
        samples = std::move(reservoir);
    }

    return samples;
}

std::pair<double, double> fit_eta_gamma(const std::vector<Sample>& samples) {
    const std::size_t n = samples.size();
    if (n < 2) {
        HFT_THROW(std::runtime_error("Insufficient samples to estimate eta/gamma"));
    }

    double sum_v = 0.0;
    double sum_s = 0.0;
    double sum_v2 = 0.0;
    double sum_vs = 0.0;

    for (const auto& sample : samples) {
        sum_v += sample.volume;
        sum_s += sample.slippage;
        sum_v2 += sample.volume * sample.volume;
        sum_vs += sample.volume * sample.slippage;
    }

    const double denom = (static_cast<double>(n) * sum_v2) - (sum_v * sum_v);
    if (std::fabs(denom) < 1e-12) {
        HFT_THROW(std::runtime_error("Degenerate system encountered while estimating eta/gamma"));
    }

    const double slope = (static_cast<double>(n) * sum_vs - sum_v * sum_s) / denom;
    const double intercept = (sum_s - slope * sum_v) / static_cast<double>(n);
    double eta = intercept;
    double gamma = 2.0 * slope;

    if (gamma < 0.0) {
        gamma = 0.0;
        eta = sum_s / static_cast<double>(n);
    }

    if (eta < 0.0) {
        eta = 0.0;
    }
    return {eta, gamma};
}

std::pair<double, double> fit_kappa_delta(
    const std::vector<Sample>& samples,
    const std::vector<double>& baseline_pred,
    const Options& opts,
    std::size_t& fitted_count
) {
    const std::array<double, 6> delta_grid{0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> ratios(samples.size(), 0.0);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        double ratio = 0.0;
        if (sample.depth > opts.min_depth) {
            ratio = sample.volume / sample.depth;
            if (!std::isfinite(ratio) || ratio <= opts.min_relative_size) {
                ratio = 0.0;
            }
        }
        ratios[i] = ratio;
    }

    double best_delta = 0.5;
    double best_kappa = 0.0;
    double best_sse = std::numeric_limits<double>::infinity();
    std::size_t best_count = 0;

    for (const double delta : delta_grid) {
        double numerator = 0.0;
        double denominator = 0.0;
        std::size_t count = 0;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double ratio = ratios[i];
            if (ratio <= 0.0) {
                continue;
            }
            const double x = std::pow(ratio, delta);
            const double residual = samples[i].slippage - baseline_pred[i];
            if (residual <= opts.min_positive_residual) {
                continue;
            }
            numerator += x * residual;
            denominator += x * x;
            ++count;
        }

        if (denominator <= 0.0) {
            continue;
        }

        double kappa = numerator / denominator;
        if (kappa < 0.0) {
            kappa = 0.0;
        }

        double sse = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double ratio = ratios[i];
            const double x = (ratio > 0.0) ? std::pow(ratio, delta) : 0.0;
            const double residual = samples[i].slippage - baseline_pred[i];
            const double diff = residual - kappa * x;
            sse += diff * diff;
        }

        if (sse < best_sse) {
            best_sse = sse;
            best_delta = delta;
            best_kappa = kappa;
            best_count = count;
        }
    }

    fitted_count = best_count;
    return {best_kappa, best_delta};
}

Metrics compute_metrics(
    const std::vector<Sample>& samples,
    const std::vector<double>& predictions
) {
    const std::size_t n = samples.size();
    if (predictions.size() != n) {
        HFT_THROW(std::runtime_error("Prediction vector size mismatch"));
    }

    double sum_sq_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_obs = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double obs = samples[i].slippage;
        const double pred = predictions[i];

        const double diff = obs - pred;
        sum_sq_error += diff * diff;
        sum_abs_error += std::fabs(diff);
        sum_obs += obs;
    }

    const double mse = sum_sq_error / static_cast<double>(n);
    const double rmse = std::sqrt(mse);
    const double mae = sum_abs_error / static_cast<double>(n);

    double sum_sq_total = 0.0;
    const double mean_obs = sum_obs / static_cast<double>(n);
    for (const auto& sample : samples) {
        const double diff = sample.slippage - mean_obs;
        sum_sq_total += diff * diff;
    }

    double r2 = std::numeric_limits<double>::quiet_NaN();
    if (sum_sq_total > 0.0) {
        r2 = 1.0 - (sum_sq_error / sum_sq_total);
    }

    Metrics metrics;
    metrics.rmse = rmse;
    metrics.mae = mae;
    metrics.r2 = r2;
    metrics.mse = mse;
    return metrics;
}

void ensure_directory(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        HFT_THROW(std::runtime_error("Failed to create directory: " + path.string()));
    }
}

void write_samples_csv(
    const std::filesystem::path& path,
    const std::vector<Sample>& samples,
    const std::vector<double>& baseline_pred,
    const std::vector<double>& extended_pred
) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Unable to write samples CSV: " + path.string()));
    }
    out << std::setprecision(12);
    out << "volume,slippage,reference_price,depth,baseline_pred,extended_pred,"
           "baseline_residual,extended_residual\n";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const double baseline_residual = samples[i].slippage - baseline_pred[i];
        const double extended_residual = samples[i].slippage - extended_pred[i];
        out << samples[i].volume << ','
            << samples[i].slippage << ','
            << samples[i].reference_price << ','
            << samples[i].depth << ','
            << baseline_pred[i] << ','
            << extended_pred[i] << ','
            << baseline_residual << ','
            << extended_residual << '\n';
    }
}

void write_json(
    const std::filesystem::path& path,
    const std::string& symbol,
    const Options& opts,
    std::size_t sample_count,
    std::size_t liquidity_points,
    double eta,
    double gamma,
    double kappa,
    double delta,
    const Metrics& baseline,
    const Metrics& extended
) {
    std::ofstream out(path);
    if (!out) {
        HFT_THROW(std::runtime_error("Unable to write calibration JSON: " + path.string()));
    }
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"symbol\": \"" << symbol << "\",\n";
    out << "  \"input_path\": \"" << opts.input_path.string() << "\",\n";
    out << "  \"sample_count\": " << sample_count << ",\n";
    out << "  \"liquidity_regression_count\": " << liquidity_points << ",\n";
    out << "  \"settings\": {\n";
    out << "    \"lookback_trades\": " << opts.lookback_trades << ",\n";
    out << "    \"lookahead_trades\": " << opts.lookahead_trades << ",\n";
    out << "    \"depth_window_ms\": " << opts.depth_window_ms << ",\n";
    out << "    \"min_abs_qty\": " << opts.min_abs_qty << ",\n";
    out << "    \"min_depth\": " << opts.min_depth << ",\n";
    out << "    \"min_positive_residual\": " << opts.min_positive_residual << ",\n";
    out << "    \"min_relative_size\": " << opts.min_relative_size << ",\n";
    if (opts.max_samples > 0) {
        out << "    \"max_samples\": " << opts.max_samples << ",\n";
        out << "    \"sample_seed\": " << opts.sample_seed << "\n";
    } else {
        out << "    \"max_samples\": null,\n";
        out << "    \"sample_seed\": null\n";
    }
    out << "  },\n";
    out << "  \"parameters\": {\n";
    out << "    \"eta\": " << eta << ",\n";
    out << "    \"gamma\": " << gamma << ",\n";
    out << "    \"kappa\": " << kappa << ",\n";
    out << "    \"delta\": " << delta << "\n";
    out << "  },\n";
    out << "  \"metrics\": {\n";
    out << "    \"baseline\": {\n";
    out << "      \"rmse\": " << baseline.rmse << ",\n";
    out << "      \"mae\": " << baseline.mae << ",\n";
    out << "      \"r2\": " << baseline.r2 << ",\n";
    out << "      \"mse\": " << baseline.mse << "\n";
    out << "    },\n";
    out << "    \"extended\": {\n";
    out << "      \"rmse\": " << extended.rmse << ",\n";
    out << "      \"mae\": " << extended.mae << ",\n";
    out << "      \"r2\": " << extended.r2 << ",\n";
    out << "      \"mse\": " << extended.mse << "\n";
    out << "    }\n";
    out << "  }\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    Options opts = parse_arguments(argc, argv);
    if (opts.input_path.empty()) {
        std::cerr << "impact_calibration requires --input PATH\n";
        return 1;
    }

    std::string symbol;
    const auto trades = load_trades(opts, symbol);
    if (trades.size() < (opts.lookback_trades + opts.lookahead_trades + 10)) {
        std::cerr << "Insufficient trades to perform calibration\n";
        return 1;
    }

    const auto depths = compute_depths(trades, opts.depth_window_ms);
    auto samples = build_samples(trades, depths, opts);
    if (samples.size() < 100) {
        std::cerr << "Collected only " << samples.size()
                  << " samples – increase horizon or relax filters\n";
        return 1;
    }

    const auto [eta, gamma] = fit_eta_gamma(samples);

    std::vector<double> baseline_pred;
    baseline_pred.reserve(samples.size());
    for (const auto& sample : samples) {
        baseline_pred.push_back(eta + 0.5 * gamma * sample.volume);
    }
    const Metrics baseline_metrics = compute_metrics(samples, baseline_pred);

    std::size_t liquidity_points = 0;
    const auto [kappa, delta] = fit_kappa_delta(samples, baseline_pred, opts, liquidity_points);

    std::vector<double> extended_pred;
    extended_pred.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        double value = baseline_pred[i];
        if (kappa > 0.0 && sample.depth > opts.min_depth) {
            const double ratio = sample.volume / sample.depth;
            if (ratio > opts.min_relative_size) {
                const double increment = (delta == 0.0) ? kappa : kappa * std::pow(ratio, delta);
                if (std::isfinite(increment)) {
                    value += increment;
                }
            }
        }
        extended_pred.push_back(value);
    }
    const Metrics extended_metrics = compute_metrics(samples, extended_pred);

    ensure_directory(opts.output_dir);
    const auto output_path = opts.output_dir / (symbol + "_calibration.json");
    const auto samples_path = opts.output_dir / (symbol + "_samples.csv");
    write_samples_csv(samples_path, samples, baseline_pred, extended_pred);
    write_json(output_path, symbol, opts, samples.size(), liquidity_points, eta, gamma, kappa, delta, baseline_metrics, extended_metrics);

    std::cout << std::setprecision(6)
              << "Calibrated impact model for " << symbol << '\n'
              << "  eta      = " << eta << '\n'
              << "  gamma    = " << gamma << '\n'
              << "  kappa    = " << kappa << '\n'
              << "  delta    = " << delta << '\n'
              << "  samples  = " << samples.size() << '\n'
              << "  liquidity regression points = " << liquidity_points << '\n'
              << "Baseline RMSE: " << baseline_metrics.rmse
              << " | Extended RMSE: " << extended_metrics.rmse << '\n';

    return 0;
}
