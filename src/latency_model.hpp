#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace simulator {

class LatencyModel {
public:
    virtual ~LatencyModel() = default;
    virtual double sample_delay(std::mt19937_64& rng) const = 0;
};

class ExponentialLatencyModel final : public LatencyModel {
public:
    explicit ExponentialLatencyModel(double mean_microseconds)
        : rate_(mean_microseconds > 0.0 ? 1.0 / (mean_microseconds * 1e-6) : 0.0) {}

    double sample_delay(std::mt19937_64& rng) const override {
        if (rate_ <= 0.0) {
            return 0.0;
        }
        std::exponential_distribution<double> expo(rate_);
        return expo(rng);
    }

private:
    double rate_;
};

template <typename Payload>
class LatencyQueue {
public:
    struct Entry {
        double ready_time;
        double delay;
        Payload payload;
    };

    bool empty() const noexcept {
        return queue_.empty();
    }

    std::size_t size() const noexcept {
        return queue_.size();
    }

    void push(double ready_time, double delay, Payload payload) {
        queue_.push(Entry{ready_time, delay, std::move(payload)});
    }

    const Entry& top() const {
        return queue_.top();
    }

    Entry pop() {
        Entry entry = std::move(queue_.top());
        queue_.pop();
        return entry;
    }

    template <typename OutputIt>
    void drain_up_to(double time, OutputIt out) {
        while (!queue_.empty() && queue_.top().ready_time <= time) {
            *out++ = pop();
        }
    }

private:
    struct Comparator {
        bool operator()(const Entry& lhs, const Entry& rhs) const noexcept {
            return lhs.ready_time > rhs.ready_time;
        }
    };

    std::priority_queue<Entry, std::vector<Entry>, Comparator> queue_;
};

} // namespace simulator

