#include <atomic>
#include <iostream>
#include <random>
#include <thread>

#include <torch/torch.h>
#include <zmq.hpp>

#include "mcts.hpp"
#include "replay.hpp"

#include "gomoku/gomoku.hpp"
// #include "tictactoe/tictactoe.hpp"

bool alive{true};
enum { GEN, TRN };
using Env = Gomoku;

void run()
{
    std::cin.get();
    alive = false;
}

void update_game_stat(int game_length, float& avg_game_length, long long& game_count)
{
    game_count++;
    avg_game_length += (game_length - avg_game_length) / game_count;
}

void print_game_stat(float avg_game_length, long long game_count)
{
    std::cout << "Average game length: " << avg_game_length << " | Game count: " << game_count << std::endl;
}


int main(int argc, char const *argv[])
{
    if (argc != 4)
        throw std::runtime_error("Invalid arguments. Usage: ./generate <nthreads> <iter_budget> <verbosity>");
    
    std::thread thd(&run);

    using S = typename Env::state_type;
    using P = torch::Tensor;
    using R = torch::Tensor;

    int nthreads = atoi(argv[1]);
    int iter_budget = atoi(argv[2]);
    int verbosity = atoi(argv[3]);
    int batch_size = 16, vl = 3, c_puct = 3;

    Env& env = Env::get();
    int board_size = env.get_board_size();

    // MCTS(int nthreads_, int batch_size_, float vl_, float c_puct_, int n_res_, int channels_)
    MCTS<Env> agent(nthreads, batch_size, vl, c_puct);
    ReplayBuffer<Env> buffer(GEN, "*", "5555");

    std::random_device rd;
    std::mt19937 gen(rd());


    S state;
    P policy;
    R reward;
    bool done;

    float avg_game_length = 25;
    long long game_count = 0;
    int i;
    while (alive) {
        agent.load();
        i = 0;
        state = env.reset();

        agent.search_probs(state, iter_budget, 0); // warmup (batch statistics)

        std::cout << "Step " << i << ":" << std::endl;
        std::cout << env.to_string(state).str() << std::endl;
        while (alive) {
            i++;

            policy = agent.search_probs(state, iter_budget, verbosity);
            buffer.temporary_append(env.get_board(state), policy);

            int point;
            if (i < avg_game_length * 0.2) {
                std::cout << "Sampling action from the distribution." << std::endl;
                auto policy_ptr = policy.data<float>();
                std::discrete_distribution<> dist(policy_ptr, policy_ptr + policy.numel());
                point = dist(gen);
            } else {
                std::cout << "Selecting deterministically." << std::endl;
                point = torch::argmax(policy).item<int>();
            }

            int y = point / board_size;
            int x = point - y * board_size;
            Env::action_type action {y, x};

            char player = env.get_player(state) == 0 ? 'O' : 'X';
            std::tie(state, reward, done) = env.step(state, action);
            
            std::cout << "Step " << i << " (Player " << player << "):" << std::endl;
            auto board_stream = env.to_string(state, action);
            auto pol_stream = visualize_stream(policy);
            adjacent_display(pol_stream, board_stream);

            if (done) {
                std::cout << "Reward:\n" << reward << std::endl;
                buffer.send_reward(reward);
                agent.clear();

                update_game_stat(i, avg_game_length, game_count);
                print_game_stat(avg_game_length, game_count);
                break;
            }
        }
    }
    thd.join();

    return 0;
}