#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct CalibrationRow {
    std::string strategy;
    double order_size{};
    double aggressiveness{};
    double latency_sensitivity{};
    double realized_pnl{};
    double trade_count{};
    double avg_latency_ms{};
    double fill_rate{};
    double liquidity_pressure{};
    double inventory_volatility{};
    double inventory_turnover{};
    double market_price_volatility{};
};

struct Totals {
    double realized{};
    double alpha{};
    double slippage{};
    double impact{};
    double carry{};
    std::size_t count{};

    void accumulate(double realized_pnl,
                    double alpha_contribution,
                    double slippage_cost,
                    double impact_cost,
                    double carry_cost) {
        realized += realized_pnl;
        alpha += alpha_contribution;
        slippage += slippage_cost;
        impact += impact_cost;
        carry += carry_cost;
        ++count;
    }
};

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> tokens;
    tokens.reserve(32);
    std::string current;
    current.reserve(32);
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == '"' ) {
            in_quotes = !in_quotes;
            continue;
        }
        if (!in_quotes && ch == ',') {
            tokens.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    tokens.push_back(current);
    return tokens;
}

double parse_double(const std::vector<std::string>& tokens,
                    const std::unordered_map<std::string, std::size_t>& indices,
                    const std::string& key) {
    auto it = indices.find(key);
    if (it == indices.end()) {
        throw std::runtime_error("Missing column: " + key);
    }
    const std::string& value = tokens.at(it->second);
    if (value.empty()) {
        return 0.0;
    }
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value '" + value + "' for column " + key);
    }
}

CalibrationRow parse_row(const std::vector<std::string>& tokens,
                         const std::unordered_map<std::string, std::size_t>& indices) {
    CalibrationRow row;
    row.strategy = tokens.at(indices.at("strategy"));
    row.order_size = parse_double(tokens, indices, "order_size");
    row.aggressiveness = parse_double(tokens, indices, "aggressiveness");
    row.latency_sensitivity = parse_double(tokens, indices, "latency_sensitivity");
    row.realized_pnl = parse_double(tokens, indices, "realized_pnl");
    row.trade_count = parse_double(tokens, indices, "trade_count");
    row.avg_latency_ms = parse_double(tokens, indices, "avg_latency_ms");
    row.fill_rate = parse_double(tokens, indices, "fill_rate");
    row.liquidity_pressure = parse_double(tokens, indices, "liquidity_pressure");
    row.inventory_volatility = parse_double(tokens, indices, "inventory_volatility");
    row.inventory_turnover = parse_double(tokens, indices, "inventory_turnover");
    row.market_price_volatility = parse_double(tokens, indices, "market_price_volatility");
    return row;
}

double compute_slippage_cost(const CalibrationRow& row) {
    const double latency_factor = row.avg_latency_ms / 5.0;
    const double fill_penalty = 1.0 - row.fill_rate + 0.5 * row.latency_sensitivity;
    const double volume = row.order_size * row.trade_count;
    return volume * latency_factor * fill_penalty * 0.01;
}

double compute_impact_cost(const CalibrationRow& row) {
    const double volume = row.order_size * row.trade_count;
    return row.liquidity_pressure * row.aggressiveness * volume * 0.0005;
}

double compute_inventory_carry(const CalibrationRow& row) {
    const double exposure = row.inventory_volatility * row.inventory_turnover;
    return exposure * row.market_price_volatility * 0.5;
}

struct SummaryRow {
    std::string strategy;
    Totals totals;
};

std::vector<SummaryRow> summarise(const std::unordered_map<std::string, Totals>& data) {
    std::vector<SummaryRow> rows;
    rows.reserve(data.size());
    for (const auto& [strategy, totals] : data) {
        rows.push_back(SummaryRow{strategy, totals});
    }
    std::sort(
        rows.begin(),
        rows.end(),
        [](const SummaryRow& lhs, const SummaryRow& rhs) {
            return lhs.strategy < rhs.strategy;
        }
    );
    return rows;
}

void write_breakdown(const std::filesystem::path& output_path,
                     const std::vector<SummaryRow>& summaries,
                     const Totals& overall) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to write breakdown CSV: " + output_path.string());
    }
    out << "strategy,runs,avg_realized_pnl,avg_strategy_alpha,"
           "avg_execution_slippage,avg_execution_impact,avg_inventory_carry,identity_error\n";
    out << std::fixed << std::setprecision(6);

    auto emit_row = [&out](const std::string& label, const Totals& totals) {
        if (totals.count == 0) {
            return;
        }
        const double denom = static_cast<double>(totals.count);
        const double avg_realized = totals.realized / denom;
        const double avg_alpha = totals.alpha / denom;
        const double avg_slippage = totals.slippage / denom;
        const double avg_impact = totals.impact / denom;
        const double avg_carry = totals.carry / denom;
        const double identity_error = avg_alpha - avg_realized - avg_slippage - avg_impact - avg_carry;
        out << label << ','
            << totals.count << ','
            << avg_realized << ','
            << avg_alpha << ','
            << avg_slippage << ','
            << avg_impact << ','
            << avg_carry << ','
            << identity_error << '\n';
    };

    for (const auto& summary : summaries) {
        emit_row(summary.strategy, summary.totals);
    }
    emit_row("ALL", overall);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path input_path =
            (argc > 1) ? std::filesystem::path(argv[1])
                       : std::filesystem::path("results/week7/strategy_calibration.csv");
        const std::filesystem::path output_path =
            (argc > 2) ? std::filesystem::path(argv[2])
                       : std::filesystem::path("results/week7/pnl_breakdown.csv");

        std::ifstream input(input_path);
        if (!input.is_open()) {
            std::cerr << "Failed to open input CSV: " << input_path << '\n';
            return 1;
        }

        std::string header_line;
        if (!std::getline(input, header_line)) {
            std::cerr << "Empty calibration CSV: " << input_path << '\n';
            return 1;
        }
        const std::vector<std::string> headers = split_csv(header_line);
        std::unordered_map<std::string, std::size_t> column_index;
        column_index.reserve(headers.size());
        for (std::size_t i = 0; i < headers.size(); ++i) {
            column_index.emplace(headers[i], i);
        }

        const std::vector<std::string> required_columns = {
            "strategy",
            "order_size",
            "aggressiveness",
            "latency_sensitivity",
            "realized_pnl",
            "trade_count",
            "avg_latency_ms",
            "fill_rate",
            "liquidity_pressure",
            "inventory_volatility",
            "inventory_turnover",
            "market_price_volatility"
        };

        for (const auto& col : required_columns) {
            if (!column_index.contains(col)) {
                std::cerr << "Required column missing from CSV: " << col << '\n';
                return 1;
            }
        }

        std::unordered_map<std::string, Totals> per_strategy;
        Totals overall{};

        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            const auto tokens = split_csv(line);
            if (tokens.size() != headers.size()) {
                std::cerr << "Skipping malformed row: " << line << '\n';
                continue;
            }
            const CalibrationRow row = parse_row(tokens, column_index);
            const double slippage = compute_slippage_cost(row);
            const double impact = compute_impact_cost(row);
            const double inventory_carry = compute_inventory_carry(row);
            const double alpha = row.realized_pnl + slippage + impact + inventory_carry;

            auto& totals = per_strategy[row.strategy];
            totals.accumulate(row.realized_pnl, alpha, slippage, impact, inventory_carry);
            overall.accumulate(row.realized_pnl, alpha, slippage, impact, inventory_carry);
        }

        const auto summaries = summarise(per_strategy);
        std::filesystem::create_directories(output_path.parent_path());
        write_breakdown(output_path, summaries, overall);
        std::cout << "PnL breakdown written to " << output_path << '\n';
    } catch (const std::exception& ex) {
        std::cerr << "pnl_analysis error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
