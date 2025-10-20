#pragma once

#include "perf/ScopedTimer.hpp"

#include <filesystem>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace perf {

struct SampleStats {
    double total_microseconds{0.0};
    double min_microseconds{std::numeric_limits<double>::max()};
    double max_microseconds{0.0};
    std::size_t count{0};

    void update(double value) noexcept {
        total_microseconds += value;
        if (value < min_microseconds) {
            min_microseconds = value;
        }
        if (value > max_microseconds) {
            max_microseconds = value;
        }
        ++count;
    }

    [[nodiscard]] double average() const noexcept {
        return count > 0 ? total_microseconds / static_cast<double>(count) : 0.0;
    }
};

class Profiler {
public:
    static Profiler& instance();

    void record(std::string_view path, double microseconds);

    [[nodiscard]] std::vector<std::pair<std::string_view, SampleStats>> snapshot() const;

    void reset();

    void write_report(const std::filesystem::path& report_path,
                      const std::filesystem::path& folded_path) const;

private:
    struct Entry {
        std::string path;
        SampleStats stats;
    };

    Profiler() = default;

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

class ScopedRecord {
public:
    explicit ScopedRecord(std::string_view path)
        : path_(path),
          timer_([this](double microseconds) {
              Profiler::instance().record(path_, microseconds);
          }) {}

private:
    std::string_view path_;
    ScopedTimer timer_;
};

class StackScope {
public:
    StackScope(std::initializer_list<std::string_view> frames);

private:
    std::string path_;
    ScopedTimer timer_;
};

} // namespace perf

#define HFT_CONCAT_IMPL(a, b) a##b
#define HFT_CONCAT(a, b) HFT_CONCAT_IMPL(a, b)

#ifdef HFT_ENABLE_PROFILING
#define HFT_PROFILE_SCOPE(path_literal) \
    ::perf::ScopedRecord HFT_CONCAT(perf_scope_, __LINE__)(path_literal)
#define HFT_PROFILE_STACK(...) \
    ::perf::StackScope HFT_CONCAT(perf_stack_scope_, __LINE__)({__VA_ARGS__})
#else
#define HFT_PROFILE_SCOPE(path_literal) (void)sizeof(path_literal)
#define HFT_PROFILE_STACK(...) (void)0
#endif
