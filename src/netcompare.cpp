#include <cstdlib>
#include <fstream>
#include <sstream>

#include <torch/torch.h>

#include "gomoku/gomoku.hpp"
#include "netconfig.hpp"
#include "network.hpp"
#include "replay.hpp"

using Env = Gomoku;

enum { GEN, TRN };

int main(int argc, char const *argv[])
{
    if (argc != 2)
        throw std::runtime_error("Wrong number of arguments supplied.");

    // DO NOT change!
    std::stringstream stream;
    int max_size = 100000;
    int train_threshold = 50;
    ReplayBuffer<Env> buffer(TRN, "localhost", "5555", max_size, train_threshold);
    buffer.load("replay");
    
    Env& env = Env::get();
    int c_in = env.get_state_channels();
    int c_out = env.get_action_channels();
    int board_size = env.get_board_size();

    int n = atoi(argv[1]);
    int batch_size = 8;
    
    NetConfig& netconf = NetConfig::get(n);
    std::cout << "Channels: [";
    stream << "Channels: [";
    for (int i : netconf.resblocks()) {
        std::cout << i << ' ';
        stream << i << ' ';
    }
    std::cout << ']' << std::endl;
    stream << ']' << std::endl;
    PVNetwork net(board_size, netconf.resblocks(), c_in, c_out);
    std::cout << "Network created." << std::endl;

    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Device: CUDA" << std::endl;
    }
    net->to(device);


    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(3e-4).beta1(0.9).beta2(0.999));

    torch::Tensor state, policy, reward;
    torch::Tensor p, v;
    torch::Tensor vloss, ploss, wloss, loss;
    double avg;
    for (int epoch = 0; epoch < 10; epoch++) {
        avg = 0;
        std::cout << "<Epoch " << epoch << '>' << std::endl;
        stream << "<Epoch " << epoch << '>' << std::endl;
        int max_step = buffer.size() / batch_size;
        for (int step = 0; step < max_step; step++) {
            net->zero_grad();
            std::tie(state, policy, reward) = buffer.get_batch(batch_size);

            state = state.to(device);
            policy = policy.to(device);
            reward = reward.to(device);
            
            std::tie(p, v) = net(state);

            vloss = torch::sum(torch::pow(v - reward, 2)) / 2;
            ploss = -torch::sum(p * policy);
            wloss = torch::zeros({}).to(device);
            for (auto& param : net->parameters())
                wloss += torch::norm(param, 2);
            
            loss = vloss + ploss + wloss * 1e-4;
            loss /= batch_size;
            loss.backward();

            optimizer.step();

            avg += (loss.item<float>() - avg) / (step + 1);

            if (step % 1000 == 0) {
                std::cout << "[" << step << "/" << max_step << "] | loss: " << loss.item<float>() << std::endl;
                stream << "[" << step << "/" << max_step << "] | loss: " << loss.item<float>() << std::endl;
            }
        }
        std::cout << "[" << max_step << "/" << max_step << "] | loss: " << loss.item<float>() << std::endl;
        stream << "[" << max_step << "/" << max_step << "] | loss: " << loss.item<float>() << std::endl;
        std::cout << "Avg " << avg << std::endl;
        stream << "Avg " << avg << std::endl;
    }

    std::ofstream file;
    file.open("log" + std::to_string(n) + ".txt");
    file << stream.rdbuf();
    file.close();
    
    return 0;
}