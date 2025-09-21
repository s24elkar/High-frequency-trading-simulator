#pragma once
#include <vector>
#include <random>
#include <functional>
#include <cmath>
#include <limits>

namespace hawkes {

struct Result {
  std::vector<double> t;
  std::vector<double> v;
};

// Exponential kernel with fast state
struct ExpKernel {
  double alpha, beta; // φ(u,v)=α v e^{-βu}
  inline double phi(double u, double v) const {
    return (u >= 0.0) ? alpha * v * std::exp(-beta * u) : 0.0;
  }
  inline double decay(double S, double dt) const {
    return S * std::exp(-beta * dt);
  }
  inline double jump(double v) const { return alpha * v; }
};

// Power-law kernel (generic, monotone)
struct PowerLawKernel {
  double alpha, c, gamma; // φ(u,v)=α v (u+c)^{-γ}
  inline double phi(double u, double v) const {
    return (u >= 0.0) ? alpha * v * std::pow(u + c, -gamma) : 0.0;
  }
};

// Fast thinning for exponential kernel (scalar state)
template<class MarkSampler, class URNG=std::mt19937_64>
Result simulate_exp(double mu, const ExpKernel& K, MarkSampler&& sample_mark,
                    double T, std::uint64_t seed=42) {
  URNG rng(seed);
  std::exponential_distribution<double> expo; // will param by rate via 1/scale
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  double t = 0.0, S = 0.0;
  Result R;

  while (t < T) {
    double lam_star = mu + S;
    if (lam_star <= 0.0) break;
    double w = expo(rng) / lam_star; // Exp(rate=lam_star)
    double t_cand = t + w;
    if (t_cand > T) break;
    double S_cand = K.decay(S, w);
    double lam_tc = mu + S_cand;
    if (unif(rng) <= lam_tc / lam_star) {
      double v = sample_mark(rng);
      R.t.push_back(t_cand);
      R.v.push_back(v);
      S = S_cand + K.jump(v);
      t = t_cand;
    } else {
      S = S_cand;
      t = t_cand;
    }
  }
  return R;
}

// Generic thinning for monotone-decreasing kernels (recompute φ-sum)
template<class Kernel, class MarkSampler, class URNG=std::mt19937_64>
Result simulate_general(double mu, const Kernel& K, MarkSampler&& sample_mark,
                        double T, std::uint64_t seed=43) {
  URNG rng(seed);
  std::exponential_distribution<double> expo;
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  double t = 0.0;
  Result R;
  double S_cur = 0.0;

  auto sum_phi = [&](double t_now) {
    double s = 0.0;
    for (size_t i = 0; i < R.t.size(); ++i) {
      s += K.phi(t_now - R.t[i], R.v[i]);
    }
    return s;
  };

  while (t < T) {
    double lam_star = mu + S_cur;
    if (lam_star <= 0.0) break;
    double w = expo(rng) / lam_star;
    double t_cand = t + w;
    if (t_cand > T) break;

    double S_tc = sum_phi(t_cand);
    double lam_tc = mu + S_tc;

    if (unif(rng) <= lam_tc / lam_star) {
      double v = sample_mark(rng);
      R.t.push_back(t_cand);
      R.v.push_back(v);
      S_cur = S_tc + K.phi(0.0, v);
      t = t_cand;
    } else {
      S_cur = S_tc;
      t = t_cand;
    }
  }
  return R;
}

} // namespace hawkes
