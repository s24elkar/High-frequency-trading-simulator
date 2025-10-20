#include "perf/Profiler.hpp"

#include <algorithm>
#include <fstream>
#include <string>

namespace perf {

namespace {
std::string build_path(std::initializer_list<std::string_view> frames) {
    std::size_t total = 0;
    if (!frames.size()) {
        return {};
    }
    for (std::string_view frame : frames) {
        total += frame.size();
    }
    total += frames.size() - 1; // separators

    std::string path;
    path.reserve(total);
    bool first = true;
    for (std::string_view frame : frames) {
        if (!first) {
            path.push_back(';');
        }
        path.append(frame.data(), frame.size());
        first = false;
    }
    return path;
}
} // namespace

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::record(std::string_view path, double microseconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : entries_) {
        if (entry.path == path) {
            entry.stats.update(microseconds);
            return;
        }
    }
    Entry entry;
    entry.path.assign(path.data(), path.size());
    entry.stats.update(microseconds);
    entries_.push_back(std::move(entry));
}

std::vector<std::pair<std::string_view, SampleStats>> Profiler::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::string_view, SampleStats>> result;
    result.reserve(entries_.size());
    for (const auto& entry : entries_) {
        result.emplace_back(entry.path, entry.stats);
    }
    return result;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}

void Profiler::write_report(const std::filesystem::path& report_path,
                            const std::filesystem::path& folded_path) const {
    const auto data = snapshot();
    if (data.empty()) {
        return;
    }

    double total_microseconds = 0.0;
    for (const auto& [_, stats] : data) {
        total_microseconds += stats.total_microseconds;
    }

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.second.total_microseconds > rhs.second.total_microseconds;
              });

    const auto report_dir = report_path.parent_path();
    if (!report_dir.empty()) {
        std::filesystem::create_directories(report_dir);
    }
    const auto folded_dir = folded_path.parent_path();
    if (!folded_dir.empty()) {
        std::filesystem::create_directories(folded_dir);
    }

    std::ofstream report(report_path);
    report << "Path,Total(us),Share(%),Avg(us),Min(us),Max(us),Samples\n";
    report.setf(std::ios::fixed, std::ios::floatfield);
    report.precision(4);
    for (const auto& [path, stats] : sorted) {
        const double share = (total_microseconds > 0.0)
                                 ? (stats.total_microseconds / total_microseconds) * 100.0
                                 : 0.0;
        report << path << ','
               << stats.total_microseconds << ','
               << share << ','
               << stats.average() << ','
               << stats.min_microseconds << ','
               << stats.max_microseconds << ','
               << stats.count << '\n';
    }

    std::ofstream folded(folded_path);
    folded.setf(std::ios::fixed, std::ios::floatfield);
    folded.precision(4);
    for (const auto& [path, stats] : data) {
        folded << path << ' ' << stats.total_microseconds << '\n';
    }
}

StackScope::StackScope(std::initializer_list<std::string_view> frames)
    : path_(build_path(frames)),
      timer_([this](double microseconds) {
          Profiler::instance().record(path_, microseconds);
      }) {}

} // namespace perf
