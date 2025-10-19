#include <iostream>

#include "simulator_core.hpp"

int main() {
    using namespace simulator;

    SimulationConfig config = default_btcusdt_config();
    SimulatorCore core(config);
    const SimulationResult result = core.run();

    std::cout << "Hawkes-driven session complete\n";
    std::cout << "Duration         : " << result.horizon << " seconds\n";
    std::cout << "Trades processed : " << result.arrivals.size() << '\n';
    std::cout << "Mean intensity   : " << result.mean_intensity << " trades/s\n";
    std::cout << "Mean inter-arrival: " << result.mean_interarrival << " s\n";
    std::cout << "Inter-arrival var: " << result.variance_interarrival << " s^2\n";
    std::cout << "Arrival log      : " << config.event_log_path << '\n';
    return 0;
}
