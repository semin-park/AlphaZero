#include <atomic>
#include <iostream>
#include <thread>

#include <torch/torch.h>
#include <zmq.hpp>

#include "mcts.hpp"
#include "replay.hpp"

#include "tictactoe/tictactoe.hpp"

bool alive{true};
enum { GEN, TRN };

void run()
{
    std::cin.get();
    alive = false;
}



int main(int argc, char const *argv[])
{
    if (argc != 3)
        throw std::runtime_error("Number of threads and iteration budget required as an argument");
    
    std::thread thd(&run);

    using S = typename TicTacToe::state_type;
    using P = typename TicTacToe::policy_type;
    using R = typename TicTacToe::reward_type;

    int nthreads = atoi(argv[1]);
    int iter_budget = atoi(argv[2]);
    int verbosity = 2;
    int batch_size = 8, vl = 3, c_puct = 3, n_res = 3, channels = 3;

    TicTacToe& env = TicTacToe::get();

    // MCTS(int nthreads_, int batch_size_, float vl_, float c_puct_, int n_res_, int channels_)
    MCTS<TicTacToe> agent(nthreads, batch_size, vl, c_puct, n_res, channels);
    ReplayBuffer<TicTacToe> buffer(GEN, "*", "5555");


    S state;
    P policy;
    R reward;
    bool done;

    int i;
    while (alive) {
        agent.load();
        i = 0;
        state = env.reset();
        std::cout << "Step " << i << ":" << std::endl;
        env.print(state);
        while (alive) {
            policy = agent.search_probs(state, iter_budget, verbosity);
            buffer.temporary_append(env.get_board(state), policy);

            int point = torch::argmax(policy).item<int>();

            int y = point / 3;
            int x = point - y * 3;

            std::tie(state, reward, done) = env.step(state, {0, y, x});
            std::cout << "Step " << i << ":" << std::endl;
            env.print(state);
            if (done) {
                std::cout << "Reward:\n" << reward << std::endl;
                buffer.send_reward(reward);
                break;
            }
        }
    }
    thd.join();

    return 0;
}