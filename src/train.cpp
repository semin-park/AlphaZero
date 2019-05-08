#include <cmath>
#include <iostream>
#include <string>
#include <thread>

#include <torch/torch.h>
#include <zmq.hpp>

#include "netconfig.hpp"
#include "network.hpp"
#include "replay.hpp"
#include "util.h"

#include "gomoku/gomoku.hpp"
// #include "tictactoe/tictactoe.hpp"

bool alive{true};
enum { GEN, TRN };
using Env = Gomoku;

/* ------------------ Set batch_size!! ------------------ */

int max_size = 100000;
int train_threshold = 50;
int batch_size = 32;
int save_every = 5000;

/* ----------------------------------------------------- */
void run()
{
    std::cin.get();
    alive = false;
}

void get_data(ReplayBuffer<Env>& buffer)
{
    // asynchronously receive data from the generator
    std::cout << "Buffer handler: " << std::this_thread::get_id() << std::endl;
    while (alive)
        buffer.receive();
}


int main(int argc, char const *argv[])
{
    if (argc != 2)
        throw std::runtime_error("Specify the current step in unit of thousand. E.g., ./train 17 (=17,000)");

    NetConfig& netconf = NetConfig::get(2);
    std::cout << netconf.channels_to_string() << std::endl;

    int step = atoi(argv[1]) * 1000 + 1;
    std::cout << "Main: " << std::this_thread::get_id() << std::endl;

    std::string dir = "replay";
    ReplayBuffer<Env> buffer(TRN, "localhost", "5555", max_size, train_threshold);
    buffer.load(dir);

    std::thread thd(&run);
    std::thread worker(&get_data, std::ref(buffer));
    worker.detach();

    Env& env = Env::get();

    int c_in = env.get_state_channels();
    int c_out = env.get_action_channels();
    int board_size = env.get_board_size();
    
    auto device = torch::kCPU;

    PVNetwork net(board_size, netconf.resblocks(), c_in, c_out, true);
    std::string path = load_network(net);

    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
        net->to(device);
    }

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).beta1(0.9).beta2(0.999));

    std::cout << "Batch size: " << batch_size << " | Max size: " << max_size << std::endl;
    try {
        torch::Tensor state, policy, reward;
        torch::Tensor p, v;
        torch::Tensor vloss, ploss, wloss, loss;
        float avg_loss = 0;
        float momentum = 0.9;
        for (; ; step++) {
            if (!alive)
                break;
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

            avg_loss = loss.item<float>() * (1 - momentum) + avg_loss * momentum;
            if (std::isnan(avg_loss)) {
                printf("ERROR: loss %f | vloss %f | ploss %f | wloss %f",
                    loss.item<float>(), vloss.item<float>(), ploss.item<float>(), wloss.item<float>());
                std::cout << "Policy: \n" << policy << std::endl;
                std::cout << "p:\n" << p << std::endl;
                exit(0);
            }

            if (step % 100 == 0) {
                std::cout << "Step " << step << " | avg loss (momentum=" << momentum << "): " << avg_loss << "\r" << std::flush;
            }
            if (step % save_every == 0) {
                std::cout << std::endl;
                path = save_network(net, path);
                buffer.save(dir);
            }
        }
        std::cout << std::endl;
    } catch (std::exception& e) {
        std::cout << std::endl << e.what() << std::endl;
    }

    thd.join();
    buffer.save(dir);

    return 0;
}