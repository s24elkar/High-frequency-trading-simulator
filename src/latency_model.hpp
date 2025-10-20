#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

#ifndef HFT_LEGACY_LATENCY_QUEUE

template <typename Payload>
class LatencyQueue {
public:
    struct Entry {
        double ready_time;
        double delay;
        Payload payload;
    };

    explicit LatencyQueue(std::size_t initial_capacity = 1024)
        : buffer_(initial_capacity > 0 ? round_up_pow2(initial_capacity) : 1024),
          mask_(buffer_.size() - 1) {}

    bool empty() const noexcept {
        return heap_.empty() && ring_size() == 0;
    }

    std::size_t size() const noexcept {
        return heap_.size() + ring_size();
    }

    void push(double ready_time, double delay, Payload payload) {
        if (ring_size() == buffer_.size()) {
            grow();
        }
        buffer_[static_cast<std::size_t>(tail_ & mask_)] = Entry{
            ready_time,
            delay,
            std::move(payload)
        };
        ++tail_;
    }

    const Entry& top() {
        drain();
        return heap_.front();
    }

    Entry pop() {
        drain();
        std::pop_heap(heap_.begin(), heap_.end(), comparator_);
        Entry entry = std::move(heap_.back());
        heap_.pop_back();
        return entry;
    }

    template <typename OutputIt>
    void drain_up_to(double time, OutputIt out) {
        drain();
        while (!heap_.empty() && heap_.front().ready_time <= time) {
            std::pop_heap(heap_.begin(), heap_.end(), comparator_);
            Entry entry = std::move(heap_.back());
            heap_.pop_back();
            *out++ = std::move(entry);
        }
    }

private:
    static std::size_t round_up_pow2(std::size_t value) {
        if (value <= 1) {
            return 1;
        }
        --value;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
#if SIZE_MAX > UINT32_MAX
        value |= value >> 32;
#endif
        return value + 1;
    }

    std::size_t ring_size() const noexcept {
        return static_cast<std::size_t>(tail_ - head_);
    }

    void grow() {
        const std::size_t new_capacity = buffer_.empty() ? 1024 : buffer_.size() * 2;
        std::vector<Entry> new_buffer(new_capacity);
        const std::size_t current = ring_size();
        for (std::size_t idx = 0; idx < current; ++idx) {
            new_buffer[idx] = std::move(buffer_[static_cast<std::size_t>((head_ + idx) & mask_)]);
        }
        buffer_.swap(new_buffer);
        head_ = 0;
        tail_ = current;
        mask_ = buffer_.size() - 1;
    }

    void drain() {
        const std::size_t pending = ring_size();
        if (pending == 0) {
            return;
        }
        for (std::size_t idx = 0; idx < pending; ++idx) {
            Entry entry = std::move(buffer_[static_cast<std::size_t>(head_ & mask_)]);
            ++head_;
            heap_.push_back(std::move(entry));
            std::push_heap(heap_.begin(), heap_.end(), comparator_);
        }
    }

    struct Comparator {
        bool operator()(const Entry& lhs, const Entry& rhs) const noexcept {
            return lhs.ready_time > rhs.ready_time;
        }
    };

    std::vector<Entry> buffer_;
    std::vector<Entry> heap_;
    std::uint64_t head_{0};
    std::uint64_t tail_{0};
    std::size_t mask_{0};
    Comparator comparator_{};
};

#else

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
        Entry entry = queue_.top();
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

#endif

} // namespace simulator
