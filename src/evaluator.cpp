#include "evaluator.hpp"

#include <array>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>

#include <sys/file.h>

#include <torch/script.h>

#include "xtensor/xadapt.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"

#include "../blokus/types.hpp"
#include "mcts.hpp"

#include <chrono>
using namespace std::chrono_literals;

Evaluator::Evaluator(MCTS* mcts_, int batch_size_)
  : batch_size(batch_size_),
    alive(true),
    mcts(mcts_),
    worker(&Evaluator::run, this)
{
    input = xt::zeros<int>({batch_size,51,13,13});
    ids.reserve(batch_size);
    
    // Neural Network setup
    setup();
    
    std::cout << "Read model from ./" << model_path << std::endl;
    std::cout << "My device is: " << ((int) device == 0 ? "CPU" : "CUDA") << std::endl;
    std::cout << "Batch size is: " << batch_size << std::endl;
}

void Evaluator::setup()
{
    int fd = open("model_location.txt", O_RDONLY);
    flock(fd, LOCK_EX);
    
    std::ifstream dir("model_location.txt");
    
    if (!dir.is_open()) {
        std::cout << "\"model_location.txt\" must be in the working directory. "
        << "Furthermore, the path that \"model_location.txt\" points to must exist"
        << std::endl;
        
        std::abort();
    }
    
    std::string export_path;
    getline(dir, export_path);
    dir.close();
    
    if (export_path != model_path) {
        model_path = export_path;
        module = torch::jit::load(export_path);
        module->to(device);
    }
}

void Evaluator::run()
{
    while (alive) {
        std::unique_lock<std::mutex> lock(mut);
        start_token.wait(lock, [this]{ return !input_q.empty() || !alive; });
        if (!alive)
            break;
        
        int size = 0;
        while (!input_q.empty() && size < batch_size) {
            auto& tup = input_q.front();
            int& idx = std::get<0>(tup);
            Board& board = std::get<1>(tup);
            
            xt::view(input, size, xt::all(), xt::all(), xt::all()) = board;
            ids[size] = idx;
            
            input_q.pop();
            
            size++;
        }
        
        std::cout << "size: " << size << std::endl;
        
        xt::xtensor<float, 4> input_f = xt::cast<float>(input);
        
        torch::Tensor X = torch::from_blob(input_f.data(), {size,51,13,13}).to(device);
        
        auto output = module->forward({X}).toTuple()->elements();
        
        // Convert back to CPU for easy memcpy'ing
        std::vector<torch::Tensor> policy_vec = output[0].toTensor().to(at::kCPU).chunk(size);
        std::vector<torch::Tensor> reward_vec = output[1].toTensor().to(at::kCPU).chunk(size);
        
        for (int i = 0; i < size; i++) {
            mcts->wait_qs[ids[i]].push(_convert_n_tie(policy_vec[i], reward_vec[i]));
            mcts->evaluated_tokens[ids[i]].notify_one();
        }
    }
}

void Evaluator::close()
{
    alive = false;
    start_token.notify_one();
    
    worker.join();
}


std::tuple<Policy, Reward> Evaluator::_convert_n_tie(const torch::Tensor& policy_t,
                                                             const torch::Tensor& reward_t)
{
    Policy policy = xt::adapt(policy_t.data<float>(), conf.action_shape);
    
    auto reward_a = reward_t.accessor<float, 2>();
    Reward reward(conf.num_players);
    for (int i = 0; i < conf.num_players; i++)
        reward[i] = reward_a[0][i];
    
    return std::tie(policy, reward);
}
