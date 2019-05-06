
//  main.cpp
//  AlphaZero
//
//  Created by Semin Park on 24/01/2019.
//  Copyright © 2019 Semin Park. All rights reserved.
//

#include <iostream>
#include <unistd.h>

#include <torch/torch.h>

#include "mcts.hpp"
#include "netconfig.hpp"
#include "network.hpp"
#include "replay.hpp"
#include "util.h"

#include "gomoku/gomoku.hpp"
// #include "tictactoe/tictactoe.hpp"

using namespace std::chrono_literals;

using Env = Gomoku;
using S = typename Env::state_type;
using B = typename Env::board_type;
using R = typename Env::reward_type;
using T = torch::Tensor;

Env& env = Env::get();

Action get_action()
{
    std::cout << "Action: ";
    std::string str;
    getline(std::cin, str);

    if (str == "q" || str == "quit" || str == "exit") exit(0);
    
    char a = str[0];
    char b = str[2];
    
    int i = int(a - '0');
    int j = int(b - 'A');
    // int j = int(b - '0');
    
    std::cout << "Action: " << i << ',' << j << std::endl;
    
    return {i,j};
}

Action get_mcts_action(MCTS<Env>& agent, const State& state, int iter_budget, int verbosity)
{
    std::cout << "Starting search..." << std::endl;
    auto policy = agent.search_probs(state, iter_budget, verbosity);
    std::cout << "Done" << std::endl;

    int point = torch::argmax(policy).item<int>();

    int y = point / env.get_board_size();
    int x = point - y * env.get_board_size();
    std::cout << "MCTS action: " << y << ',' << x << std::endl;
    return {y, x};
}

int main(int argc, const char * argv[]) {
    enum { HUM, CMP };
    std::cout << "Play against? ";
    std::string response;
    getline(std::cin, response);
    bool match = false;
    if (response == "y" || response == "Y" || response == "yes")
        match = true;

    int c_in = env.get_state_channels();
    int c_out = env.get_action_channels();
    int board_size = env.get_board_size();

    if (match) {
        std::cout << "Want to play first? ";
        std::string first;
        getline(std::cin, first);
        
        MCTS<Env> agent(3, 8, 3, 3);

        int player = CMP;
        
        if (first == "y" || first == "Y" || first == "yes")
            player = HUM;

        bool done = false;
        Action action;
        R reward;
        auto state = env.reset();
        auto board_stream = env.to_string(state);
        std::cout << board_stream.str() << std::endl;
        for (int i = 1; ; i++) {
            if (player == HUM) {
                std::cout << "Human:" << std::endl;
                action = get_action();
            } else {
                std::cout << "MCTS:" << std::endl;
                action = get_mcts_action(agent, state, 1600, 3);
            }
            
            std::tie(state, reward, done) = env.step(state, action);
            
            char mark = player ? 'X' : 'O';
            std::cout << "Step " << i << " (Player " << mark << "):" << std::endl;
            auto board_stream = env.to_string(state);
            std::cout << board_stream.str() << std::endl;
            player = !player;

            if (done) {
                std::cout << "Reward:\n" << reward << std::endl;
                agent.clear();
                break;
            }
        }
    } else {
        NetConfig& netconf = NetConfig::get(2);
        std::cout << netconf.channels_to_string() << std::endl;

        PVNetwork net(board_size, netconf.resblocks(), c_in, c_out);
        std::string path = load_network(net);

        S state = env.reset();
        B board;
        R reward;
        bool done = false;

        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

        torch::Tensor p, v;
        std::cout << "Step " << 1 << std::endl;
        std::cout << "State:" << std::endl;
        std::cout << env.to_string(state).str() << std::endl;

        for (int i = 1; ; i++) {
            if (done)
                break;
            board = env.get_board(state);
            std::tie(p, v) = net(board.unsqueeze(0).to(device).to(torch::kFloat32));
            p = p.to(torch::kCPU);
            v = v.to(torch::kCPU);
            auto actions = env.possible_actions(state, env.get_player(state));
            torch::Tensor actual_policy = torch::zeros({board_size, board_size});
            for (auto& a : actions) {
                int i = a[0];
                int j = a[1];
                actual_policy[i][j] = p.squeeze()[i][j];
            }

            int point = torch::argmax(actual_policy).item<int>();

            int y = point / board_size;
            int x = point - y * board_size;

            std::tie(state, reward, done) = env.step(state, {y, x});
            std::cout << "Step " << i << std::endl;
            std::cout << "State:" << std::endl;
            std::cout << env.to_string(state).str() << std::endl;
            std::cout << "Predicted policy:\n" << actual_policy << std::endl;
            std::cout << "Predicted reward:\n" << v << std::endl << std::endl;
        }
    }

    return 0;
}



