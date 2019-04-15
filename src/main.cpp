//
//  main.cpp
//  AlphaZero
//
//  Created by Semin Park on 24/01/2019.
//  Copyright © 2019 Semin Park. All rights reserved.
//

#include <iostream>
#include <unistd.h>

#include <torch/torch.h>

#include "netconfig.hpp"
#include "network.hpp"
#include "replay.hpp"
#include "util.h"

#include "tictactoe/tictactoe.hpp"

using namespace std::chrono_literals;


int main(int argc, const char * argv[]) {
    using S = typename TicTacToe::state_type;
    using B = typename TicTacToe::board_type;
    using R = typename TicTacToe::reward_type;
    
    TicTacToe& env = TicTacToe::get();

    int c_in, c_out, board_size;
    std::tie(c_in, c_out, board_size) = env.get_shape_info();
    NetConfig& netconf = NetConfig::get();

    PVNetwork net(board_size, netconf.resblocks(), c_in, netconf.channels(), c_out);
    std::string path = load_network(net);

    S state = env.reset();
    B board;
    R reward;
    bool done = false;

    torch::Tensor p, v;

    for (int i = 1; ; i++) {
        if (done)
            break;
        board = env.get_board(state);
        std::tie(p, v) = net(board.unsqueeze(0).to(torch::kFloat32));

        auto actions = env.possible_actions(state, env.get_player(state));
        torch::Tensor actual_policy = torch::zeros({3,3});
        for (auto& a : actions) {
            int _,i,j;
            std::tie(_,i,j) = a;
            actual_policy[i][j] = p.squeeze()[i][j];
        }

        std::cout << "Step " << i << std::endl;
        std::cout << "State:" << std::endl;
        env.print(state);
        std::cout << "Predicted policy:\n" << actual_policy << std::endl;
        std::cout << "Predicted reward:\n" << v << std::endl << std::endl;

        int point = torch::argmax(actual_policy).item<int>();

        int y = point / 3;
        int x = point - y * 3;

        std::tie(state, reward, done) = env.step(state, {0, y, x});
    }

    return 0;
}










    // torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-5).beta1(0.9).beta2(0.999).weight_decay(1e-4));
    // for (int i = 1; ; i++) {
    //     std::tie(p, v) = net(board.unsqueeze(0).to(torch::kFloat32));

    //     torch::Tensor vloss = torch::sum(torch::pow(v - reward, 2)) / 2;
    //     torch::Tensor ploss = -torch::sum(torch::log(p) * policy);

    //     torch::Tensor wloss = torch::zeros({1});
    //     for (auto& param : net->parameters()) {
    //         wloss += torch::norm(param, 2);
    //     }
    //     torch::Tensor loss = vloss + ploss + 0.05 * wloss;

    //     // if (i % 10 == 0) {
    //     //     std::cout << "Step " << i << std::endl;
    //     //     std::cout << "Predicted policy:\n" << p << std::endl;
    //     //     std::cout << "Predicted reward:\n" << v << std::endl;

    //     //     std::cout << "vloss:\n" << vloss << std::endl;
    //     //     std::cout << "ploss:\n" << ploss << std::endl;
    //     //     std::cout << "wloss:\n" << wloss << std::endl;

    //     //     std::cout << "loss:\n" << loss << std::endl;
    //     //     std::cout << "learning rate:\n" << optimizer.options.learning_rate_ << std::endl;
    //     // }

    //     loss.backward();

    //     optimizer.step();
    // }


