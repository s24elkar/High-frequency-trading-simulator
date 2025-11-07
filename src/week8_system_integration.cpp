#include "spdlog/spdlog.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <cmath>
#include <sstream>

#include <sys/resource.h>
#include <unistd.h>

namespace integration {

using Clock = std::chrono::steady_clock;

struct ScenarioConfig {
    std::string name;
    std::size_t event_count;
    double event_rate_hz;
    double price_volatility;
    double signal_threshold;
    double order_size;
    double latency_mean_us;
    double latency_jitter_us;
    unsigned int seed;
};

struct MarketEvent {
    double price{0.0};
    double timestamp_s{0.0};
    std::size_t sequence{0};
};

enum class Side { Buy, Sell };

struct OrderEvent {
    std::size_t id{0};
    Side side{Side::Buy};
    double price{0.0};
    double quantity{0.0};
    double timestamp_s{0.0};
    std::string strategy;
};

struct ExecutionEvent {
    std::size_t order_id{0};
    Side side{Side::Buy};
    double fill_price{0.0};
    double quantity{0.0};
    double timestamp_s{0.0};
    double latency_us{0.0};
};

struct CpuSample {
    double user_s{0.0};
    double sys_s{0.0};
};

CpuSample sample_cpu_time() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const double user = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1'000'000.0;
    const double sys = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1'000'000.0;
    return CpuSample{user, sys};
}

double resident_memory_mb() {
    std::ifstream statm("/proc/self/statm");
    long total_pages = 0;
    long rss_pages = 0;
    if (!(statm >> total_pages >> rss_pages)) {
        return 0.0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    return static_cast<double>(rss_pages) * static_cast<double>(page_size) / (1024.0 * 1024.0);
}

std::string fmt_double(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void log_payload(const std::string& payload) {
    spdlog::info("{}", payload);
}

class TaskExecutor {
public:
    explicit TaskExecutor(std::size_t workers)
        : stopping_(false) {
        if (workers == 0) {
            workers = 2;
        }
        workers_.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i) {
            workers_.emplace_back([this](std::stop_token token) { worker_loop(token); });
        }
    }

    ~TaskExecutor() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        // timers_ threads will stop automatically upon destruction
    }

    void post(std::function<void()> task) {
        {
            std::lock_guard lock(mutex_);
            tasks_.push(std::move(task));
        }
        pending_tasks_.fetch_add(1);
        cv_.notify_one();
    }

    void wait_until_idle() {
        std::unique_lock lock(mutex_);
        idle_cv_.wait(lock, [this]() {
            return tasks_.empty() && pending_tasks_.load() == 0;
        });
    }

private:
    void worker_loop(std::stop_token token) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this, &token]() {
                    return stopping_ || !tasks_.empty() || token.stop_requested();
                });
                if ((stopping_ || token.stop_requested()) && tasks_.empty()) {
                    break;
                }
                if (tasks_.empty()) {
                    continue;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
            pending_tasks_.fetch_sub(1);
            std::lock_guard lock(mutex_);
            if (tasks_.empty() && pending_tasks_.load() == 0) {
                idle_cv_.notify_all();
            }
        }
    }

    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    bool stopping_;
    std::atomic<std::size_t> pending_tasks_{0};
};

class MetricsCollector {
public:
    explicit MetricsCollector(std::size_t expected_events)
        : expected_events_(expected_events) {}

    void start() {
        start_time_ = Clock::now();
    }

    void stop() {
        end_time_ = Clock::now();
    }

    void record_market() {
        ++market_events_;
    }

    void record_order() {
        ++orders_sent_;
    }

    void record_fill(double latency_us) {
        ++fills_;
        latencies_us_.push_back(latency_us);
    }

    [[nodiscard]] double average_latency_us() const {
        if (latencies_us_.empty()) {
            return 0.0;
        }
        const double sum = std::accumulate(latencies_us_.begin(), latencies_us_.end(), 0.0);
        return sum / static_cast<double>(latencies_us_.size());
    }

    [[nodiscard]] double throughput_events_per_sec() const {
        const double total_events = static_cast<double>(market_events_ + orders_sent_ + fills_);
        const double elapsed = elapsed_seconds();
        if (elapsed <= 0.0) {
            return 0.0;
        }
        return total_events / elapsed;
    }

    [[nodiscard]] double elapsed_seconds() const {
        if (end_time_ <= start_time_) {
            return 0.0;
        }
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }

    [[nodiscard]] Clock::time_point start_time() const {
        return start_time_;
    }

    [[nodiscard]] std::size_t market_events() const { return market_events_; }

private:
    std::size_t expected_events_;
    std::size_t market_events_{0};
    std::size_t orders_sent_{0};
    std::size_t fills_{0};
    Clock::time_point start_time_{Clock::now()};
    Clock::time_point end_time_{Clock::now()};
    std::vector<double> latencies_us_;
};

class EventBus {
public:
    explicit EventBus(TaskExecutor& executor)
        : executor_(executor) {}

    using MarketHandler = std::function<void(const MarketEvent&)>;
    using OrderHandler = std::function<void(const OrderEvent&)>;
    using ExecutionHandlerFn = std::function<void(const ExecutionEvent&)>;

    void subscribe_market(MarketHandler handler) {
        market_handlers_.push_back(std::move(handler));
    }

    void subscribe_orders(OrderHandler handler) {
        order_handlers_.push_back(std::move(handler));
    }

    void subscribe_executions(ExecutionHandlerFn handler) {
        execution_handlers_.push_back(std::move(handler));
    }

    void publish_market(const MarketEvent& event) {
        for (const auto& handler : market_handlers_) {
            executor_.post([handler, event]() { handler(event); });
        }
    }

    void publish_order(const OrderEvent& event) {
        for (const auto& handler : order_handlers_) {
            executor_.post([handler, event]() { handler(event); });
        }
    }

    void publish_execution(const ExecutionEvent& event) {
        for (const auto& handler : execution_handlers_) {
            executor_.post([handler, event]() { handler(event); });
        }
    }

private:
    TaskExecutor& executor_;
    std::vector<MarketHandler> market_handlers_;
    std::vector<OrderHandler> order_handlers_;
    std::vector<ExecutionHandlerFn> execution_handlers_;
};

class MarketDataSimulator {
public:
    MarketDataSimulator(
        TaskExecutor& executor,
        EventBus& bus,
        MetricsCollector& metrics,
        ScenarioConfig config
    )
        : executor_(executor),
          bus_(bus),
          metrics_(metrics),
          config_(std::move(config)),
          rng_(config_.seed),
          noise_(0.0, config_.price_volatility),
          interval_(std::chrono::microseconds(
              static_cast<int>(std::max(200.0, std::round(1'000'000.0 / config_.event_rate_hz))))) {}

    void start() {
        worker_ = std::jthread([this](std::stop_token token) { run(token); });
    }

    void join() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run(std::stop_token token) {
        for (std::size_t i = 0; i < config_.event_count && !token.stop_requested(); ++i) {
            std::this_thread::sleep_for(interval_);
            emit_tick(i + 1);
        }
    }

    void emit_tick(std::size_t sequence) {
        last_price_ = std::max(1.0, last_price_ + noise_(rng_));
        metrics_.record_market();
        const auto now = Clock::now();
        const double timestamp = std::chrono::duration<double>(now - metrics_.start_time()).count();
        MarketEvent event{last_price_, timestamp, sequence};
        bus_.publish_market(event);
        if (sequence % 250 == 0) {
            std::ostringstream oss;
            oss << "{\"event\":\"market_tick\",\"sequence\":" << sequence
                << ",\"price\":" << fmt_double(event.price, 4) << "}";
            log_payload(oss.str());
        }
    }

    TaskExecutor& executor_;
    EventBus& bus_;
    MetricsCollector& metrics_;
    ScenarioConfig config_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> noise_;
    std::chrono::microseconds interval_;
    double last_price_{100.0};
    std::jthread worker_;
};

class StrategyEngine : public std::enable_shared_from_this<StrategyEngine> {
public:
    StrategyEngine(EventBus& bus, MetricsCollector& metrics, const ScenarioConfig& config, std::string name)
        : bus_(bus),
          metrics_(metrics),
          alpha_(0.05),
          threshold_(config.signal_threshold),
          order_size_(config.order_size),
          strategy_name_(std::move(name)) {}

    void start() {
        bus_.subscribe_market([self = shared_from_this()](const MarketEvent& event) {
            self->on_market(event);
        });
    }

private:
    void on_market(const MarketEvent& event) {
        if (!initialised_) {
            ema_ = event.price;
            initialised_ = true;
        }
        ema_ = ema_ * (1.0 - alpha_) + alpha_ * event.price;
        const double signal = event.price - ema_;
        if (std::abs(signal) < threshold_) {
            return;
        }
        OrderEvent order;
        order.id = next_order_id_++;
        order.side = signal > 0 ? Side::Sell : Side::Buy;
        order.strategy = strategy_name_;
        order.quantity = order_size_;
        order.price = event.price + (signal > 0 ? 0.02 : -0.02);
        order.timestamp_s = event.timestamp_s;
        metrics_.record_order();
        bus_.publish_order(order);
        const char* side_label = signal > 0 ? "SELL" : "BUY";
        std::ostringstream oss;
        oss << "{\"event\":\"order\",\"strategy\":\"" << order.strategy
            << "\",\"order_id\":" << order.id
            << ",\"side\":\"" << side_label << "\""
            << ",\"price\":" << fmt_double(order.price, 4)
            << ",\"qty\":" << fmt_double(order.quantity, 2) << "}";
        log_payload(oss.str());
    }

    EventBus& bus_;
    MetricsCollector& metrics_;
    double alpha_;
    double threshold_;
    double order_size_;
    std::string strategy_name_;
    double ema_{0.0};
    bool initialised_{false};
    std::size_t next_order_id_{1};
};

class ExecutionHandler : public std::enable_shared_from_this<ExecutionHandler> {
public:
    ExecutionHandler(
        TaskExecutor& executor,
        EventBus& bus,
        MetricsCollector& metrics,
        const ScenarioConfig& config
    )
        : executor_(executor),
          bus_(bus),
          metrics_(metrics),
          rng_(config.seed + 99),
          latency_mean_(config.latency_mean_us),
          latency_jitter_(config.latency_jitter_us),
          slippage_dist_(0.0, 0.01) {}

    void start() {
        bus_.subscribe_orders([self = shared_from_this()](const OrderEvent& order) {
            self->on_order(order);
        });
    }

private:
    void on_order(const OrderEvent& order) {
        const double latency = sample_latency();
        executor_.post([this, order, latency]() { fill_order(order, latency); });
    }

    void fill_order(const OrderEvent& order, double latency) {
        const double direction = order.side == Side::Buy ? 1.0 : -1.0;
        const double slippage = direction * slippage_dist_(rng_);
        ExecutionEvent exec;
        exec.order_id = order.id;
        exec.side = order.side;
        exec.quantity = order.quantity;
        exec.fill_price = std::max(0.5, order.price + slippage);
        exec.latency_us = latency;
        exec.timestamp_s = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
        metrics_.record_fill(latency);
        bus_.publish_execution(exec);
        std::ostringstream oss;
        oss << "{\"event\":\"execution\",\"order_id\":" << exec.order_id
            << ",\"price\":" << fmt_double(exec.fill_price, 4)
            << ",\"qty\":" << fmt_double(exec.quantity, 2)
            << ",\"latency_us\":" << fmt_double(exec.latency_us, 2) << "}";
        log_payload(oss.str());
    }

    double sample_latency() {
        std::normal_distribution<double> latency(latency_mean_, latency_jitter_);
        return std::max(25.0, latency(rng_));
    }

    TaskExecutor& executor_;
    EventBus& bus_;
    MetricsCollector& metrics_;
    std::mt19937_64 rng_;
    double latency_mean_;
    double latency_jitter_;
    std::normal_distribution<double> slippage_dist_;
};

class RiskAnalytics : public std::enable_shared_from_this<RiskAnalytics> {
public:
    explicit RiskAnalytics(EventBus& bus)
        : bus_(bus) {}

    void start() {
        bus_.subscribe_market([self = shared_from_this()](const MarketEvent& event) {
            self->on_market(event);
        });
        bus_.subscribe_executions([self = shared_from_this()](const ExecutionEvent& exec) {
            self->on_execution(exec);
        });
    }

private:
    void on_market(const MarketEvent& event) {
        last_price_ = event.price;
        update_unrealized();
    }

    void on_execution(const ExecutionEvent& exec) {
        const double signed_qty = (exec.side == Side::Buy ? 1.0 : -1.0) * exec.quantity;
        double remaining = signed_qty;
        auto opposite_sign = [](double a, double b) {
            return (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0);
        };
        while (!inventory_lots_.empty() && std::abs(remaining) > 1e-9 &&
               opposite_sign(remaining, inventory_lots_.front().size)) {
            auto& lot = inventory_lots_.front();
            const double matched = std::min(std::abs(remaining), std::abs(lot.size));
            const double lot_sign = lot.size > 0 ? 1.0 : -1.0;
            realized_pnl_ += matched * (exec.fill_price - lot.price) * lot_sign;
            lot.size -= matched * lot_sign;
            remaining -= matched;
            if (std::abs(lot.size) <= 1e-9) {
                inventory_lots_.erase(inventory_lots_.begin());
            }
        }
        if (std::abs(remaining) > 1e-9) {
            inventory_lots_.push_back({remaining, exec.fill_price});
        }
        position_ += signed_qty;
        update_unrealized();
        std::ostringstream oss;
        oss << "{\"event\":\"pnl_update\",\"position\":" << fmt_double(position_, 2)
            << ",\"realized\":" << fmt_double(realized_pnl_, 2)
            << ",\"unrealized\":" << fmt_double(unrealized_pnl_, 2) << "}";
        log_payload(oss.str());
    }

    void update_unrealized() {
        if (std::abs(position_) <= 1e-9) {
            unrealized_pnl_ = 0.0;
            return;
        }
        double total_cost = 0.0;
        double total_qty = 0.0;
        for (const auto& lot : inventory_lots_) {
            total_cost += lot.price * lot.size;
            total_qty += lot.size;
        }
        if (std::abs(total_qty) <= 1e-9) {
            unrealized_pnl_ = 0.0;
            return;
        }
        const double avg_cost = total_cost / total_qty;
        unrealized_pnl_ = position_ * (last_price_ - avg_cost);
    }

    struct Lot {
        double size;
        double price;
    };

    EventBus& bus_;
    double position_{0.0};
    double realized_pnl_{0.0};
    double unrealized_pnl_{0.0};
    double last_price_{100.0};
    std::vector<Lot> inventory_lots_;
};

struct ScenarioResult {
    std::string name;
    std::size_t event_count;
    double event_rate_hz;
    double avg_latency_us;
    double throughput_eps;
    double cpu_percent;
    double memory_mb;
};

class ScenarioRunner {
public:
    ScenarioRunner(ScenarioConfig config)
        : config_(std::move(config)) {}

    ScenarioResult run() {
        {
            std::ostringstream oss;
            oss << "{\"event\":\"scenario_start\",\"name\":\"" << config_.name
                << "\",\"events\":" << config_.event_count
                << ",\"event_rate_hz\":" << fmt_double(config_.event_rate_hz, 2) << "}";
            log_payload(oss.str());
        }

        TaskExecutor executor(std::max(2u, std::thread::hardware_concurrency()));
        EventBus bus(executor);
        MetricsCollector metrics(config_.event_count);
        metrics.start();

        auto market = std::make_unique<MarketDataSimulator>(executor, bus, metrics, config_);
        auto strategy = std::make_shared<StrategyEngine>(bus, metrics, config_, "integration_strategy");
        auto exec = std::make_shared<ExecutionHandler>(executor, bus, metrics, config_);
        auto risk = std::make_shared<RiskAnalytics>(bus);

        market->start();
        strategy->start();
        exec->start();
        risk->start();

        CpuSample cpu_before = sample_cpu_time();
        market->join();
        executor.wait_until_idle();
        metrics.stop();
        CpuSample cpu_after = sample_cpu_time();

        const double wall = metrics.elapsed_seconds();
        double cpu_total = (cpu_after.user_s - cpu_before.user_s) + (cpu_after.sys_s - cpu_before.sys_s);
        double cpu_percent = (wall > 0.0) ? (cpu_total / wall) * 100.0 : 0.0;
        const double memory = resident_memory_mb();

        ScenarioResult result{
            config_.name,
            metrics.market_events(),
            config_.event_rate_hz,
            metrics.average_latency_us(),
            metrics.throughput_events_per_sec(),
            cpu_percent,
            memory
        };

        {
            std::ostringstream oss;
            oss << "{\"event\":\"scenario_complete\",\"name\":\"" << result.name
                << "\",\"avg_latency_us\":" << fmt_double(result.avg_latency_us, 2)
                << ",\"throughput_eps\":" << fmt_double(result.throughput_eps, 2)
                << ",\"cpu_percent\":" << fmt_double(result.cpu_percent, 2)
                << ",\"mem_mb\":" << fmt_double(result.memory_mb, 2) << "}";
            log_payload(oss.str());
        }
        spdlog::flush();
        return result;
    }

private:
    ScenarioConfig config_;
};

void write_results(const std::filesystem::path& path, const std::vector<ScenarioResult>& results) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "scenario,event_rate_hz,event_count,avg_order_latency_us,event_throughput_eps,cpu_percent,memory_mb\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(3);
    for (const auto& result : results) {
        out << result.name << ','
            << result.event_rate_hz << ','
            << result.event_count << ','
            << result.avg_latency_us << ','
            << result.throughput_eps << ','
            << result.cpu_percent << ','
            << result.memory_mb << '\n';
    }
}

} // namespace integration

int main() {
    using namespace integration;
    const std::filesystem::path log_path = std::filesystem::path("logs/week8/integration_test.log");
    const auto logger = spdlog::basic_logger_mt("week8_integration", log_path);
    spdlog::set_default_logger(logger);

    std::vector<ScenarioConfig> scenarios = {
        {"baseline", 1200, 1500.0, 0.05, 0.08, 2.0, 250.0, 50.0, 42},
        {"high_load", 2500, 3200.0, 0.08, 0.06, 2.5, 320.0, 80.0, 1337},
        {"stress", 4000, 5500.0, 0.12, 0.05, 3.0, 380.0, 120.0, 9001}
    };

    std::vector<ScenarioResult> results;
    results.reserve(scenarios.size());
    for (const auto& scenario : scenarios) {
        ScenarioRunner runner(scenario);
        results.push_back(runner.run());
    }

    write_results("results/week8/system_perf.csv", results);
    {
        std::ostringstream oss;
        oss << "{\"event\":\"integration_summary\",\"scenarios\":" << results.size()
            << ",\"output\":\"results/week8/system_perf.csv\"}";
        log_payload(oss.str());
    }
    spdlog::flush();
    return 0;
}
