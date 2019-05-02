#include <iostream>
#include <string>
#include <thread>

#include <torch/torch.h>

#include "netconfig.hpp"
#include "network.hpp"
#include "util.h"

#include "gomoku/gomoku.hpp"

using Env = Gomoku;

int main(int argc, char const *argv[])
{
    if (!torch::cuda::is_available()) {
        std::cout << "You need to run this in an CUDA enabled environment" << std::endl;
        return 0;
    }

    if (argc != 2)
        throw std::runtime_error("Specify the network path you're trying to convert.");
    std::string path = argv[1];

    Env& env = Env::get();

    int c_in = env.get_state_channels();
    int c_out = env.get_action_channels();
    int board_size = env.get_board_size();

    NetConfig& netconf = NetConfig::get(0);

    PVNetwork net(board_size, netconf.resblocks(), c_in, c_out);
    std::cout << "Loading Network." << std::endl;
    torch::load(net, path);
    std::cout << "Network loaded." << std::endl;

    net->to(torch::kCPU);

    int idx = path.find("_");
    auto dot = path.find(".") - 1;
    auto diff = dot - idx;

    auto ver = path.substr(idx+1, diff);
    int version = std::atoi(ver.c_str());

    std::string cpu_path = path.substr(0, idx + 1) + "CPU_" + std::to_string(version) + ".pt";

    std::cout << "Saving CPU Network." << std::endl;
    torch::save(net, cpu_path);
    std::cout << "Network saved." << std::endl;
    return 0;
}
