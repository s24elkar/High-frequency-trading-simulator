#include "hawkes.hpp"
#include <random>
#include <iostream>

int main() {
  using namespace hawkes;

  // Exponential demo
  ExpKernel Kexp{0.8, 1.2};
  auto mark_lognormal = [](auto& rng){
    std::lognormal_distribution<double> d(0.0, 0.5);
    return d(rng);
  };
  auto R1 = simulate_exp(0.2, Kexp, mark_lognormal, 200.0, 123);

  std::cout << "[EXP] generated " << R1.t.size() << " events\n";

  // Power-law demo
  PowerLawKernel Kpl{0.1, 0.1, 1.4};
  auto mark_exp = [](auto& rng){
    std::exponential_distribution<double> d(1.0);
    return d(rng);
  };
  auto R2 = simulate_general(0.2, Kpl, mark_exp, 50.0, 456);
  std::cout << "[PL]  generated " << R2.t.size() << " events\n";
}
