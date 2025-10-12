#pragma once

#include <chrono>
#include <functional>
#include <utility>

namespace perf {

class ScopedTimer {
 public:
  using Callback = std::function<void(double)>;

  explicit ScopedTimer(Callback callback)
      : start_(std::chrono::steady_clock::now()), callback_(std::move(callback)) {}

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  ScopedTimer(ScopedTimer&&) = delete;
  ScopedTimer& operator=(ScopedTimer&&) = delete;

  ~ScopedTimer() {
    if (!callback_) {
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::micro>(end - start_);
    callback_(elapsed.count());
  }

 private:
  std::chrono::steady_clock::time_point start_;
  Callback callback_;
};

}  // namespace perf
