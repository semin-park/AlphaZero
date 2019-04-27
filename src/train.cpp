#include <iostream>
#include <string>
#include <thread>

#include <torch/torch.h>
#include <zmq.hpp>

#include "netconfig.hpp"
#include "network.hpp"
#include "replay.hpp"
#include "util.h"

#include "tictactoe/tictactoe.hpp"

bool alive{true};
enum { GEN, TRN };

/* ------------------ Set batch_size!! ------------------ */

int max_size = 100000;
int train_threshold = 50;
ReplayBuffer<TicTacToe> buffer(TRN, "localhost", "5555", max_size, train_threshold);
int batch_size = 32;

/* ----------------------------------------------------- */


void run()
{
    std::cin.get();
    alive = false;
}

void get_data()
{
    // asynchronously receive data from the generator
    while (alive)
        buffer.receive();
}


int main(int argc, char const *argv[])
{
    std::thread thd(&run);
    std::thread worker(&get_data);
    worker.detach();

    TicTacToe& env = TicTacToe::get();

    int c_in, c_out, board_size;
    std::tie(c_in, c_out, board_size) = env.get_shape_info();
    NetConfig& netconf = NetConfig::get();

    PVNetwork net(board_size, netconf.resblocks(), c_in, netconf.channels(), c_out);
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA" << std::endl;
        net->to(torch::kCUDA);
    }

    std::string path = load_network(net);
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-5).beta1(0.9).beta2(0.999));

    std::string dir = "replay";
    buffer.load(dir);

    std::cout << "Batch size: " << batch_size << " | Max size: " << max_size << std::endl;
    try {
        torch::Tensor state, policy, reward;
        torch::Tensor p, v;
        for (int i = 1; ; i++) {
            if (!alive)
                break;

            net->zero_grad();
            std::tie(state, policy, reward) = buffer.get_batch(batch_size);

            
            std::tie(p, v) = net(state);

            torch::Tensor vloss = torch::sum(torch::pow(v - reward, 2)) / 2;
            torch::Tensor ploss = -torch::sum(torch::log(p) * policy);
            torch::Tensor wloss = torch::zeros({});
            for (auto& param : net->parameters())
                wloss += torch::norm(param, 2);
            
            torch::Tensor loss = vloss + ploss + wloss * 0.1;
            loss /= batch_size;
            loss.backward();

            optimizer.step();

            if (i % 100 == 0)
                std::cout << "Step " << i << " | loss: " << loss.item<float>() << "\r" << std::flush;
            if (i % 1000 == 0)
                path = save_network(net, path);
        }
        std::cout << std::endl;
    } catch (std::exception& e) {
        std::cout << std::endl << e.what() << std::endl;
    }

    thd.join();
    buffer.save(dir);

    return 0;
}