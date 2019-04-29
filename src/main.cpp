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

#include "gomoku/gomoku.hpp"
// #include "tictactoe/tictactoe.hpp"

using namespace std::chrono_literals;

using Env = Gomoku;


int main(int argc, const char * argv[]) {
    using S = typename Env::state_type;
    using B = typename Env::board_type;
    using R = typename Env::reward_type;
    
    Env& env = Env::get();

    int c_in = env.get_state_channels();
    int c_out = env.get_action_channels();
    int board_size = env.get_board_size();
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
        torch::Tensor actual_policy = torch::zeros({board_size, board_size});
        for (auto& a : actions) {
            int i = a[0];
            int j = a[1];
            actual_policy[i][j] = p.squeeze()[i][j];
        }

        std::cout << "Step " << i << std::endl;
        std::cout << "State:" << std::endl;
        std::cout << env.to_string(state).str() << std::endl;
        std::cout << "Predicted policy:\n" << actual_policy << std::endl;
        std::cout << "Predicted reward:\n" << v << std::endl << std::endl;

        int point = torch::argmax(actual_policy).item<int>();

        int y = point / 3;
        int x = point - y * 3;

        std::tie(state, reward, done) = env.step(state, {y, x});
    }

    return 0;
}



