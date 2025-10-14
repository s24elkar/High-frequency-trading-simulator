#pragma once

#include <algorithm>
#include <cmath>

namespace Catch {

class Approx {
public:
    explicit Approx(double value) : value_(value) {}

    Approx& margin(double margin) {
        margin_ = margin;
        return *this;
    }

    Approx& epsilon(double epsilon) {
        epsilon_ = epsilon;
        return *this;
    }

    friend bool operator==(double lhs, const Approx& rhs) {
        const double diff = std::fabs(lhs - rhs.value_);
        const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs.value_)});
        return diff <= rhs.margin_ + rhs.epsilon_ * scale;
    }

    friend bool operator==(const Approx& lhs, double rhs) {
        return rhs == lhs;
    }

    friend bool operator!=(double lhs, const Approx& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const Approx& lhs, double rhs) {
        return !(lhs == rhs);
    }

private:
    double value_;
    double margin_{0.0};
    double epsilon_{1e-9};
};

} // namespace Catch

