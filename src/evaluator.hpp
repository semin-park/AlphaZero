#ifndef evaluator_hpp
#define evaluator_hpp

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/script.h>

#ifdef _CUDA
#include <ATen/cuda/CUDAContext.h>
#endif // _CUDA

#include "../blokus/config.hpp"
#include "../blokus/types.hpp"

class MCTS;

class Evaluator {

friend class MCTS;
    
public:
    static Evaluator& get(MCTS* mcts_, int batch_size_)
    {
        static Evaluator eval(mcts_, batch_size_);
        return eval;
    }
    
    void run();
    
    void close();
    
private:
    Evaluator(MCTS* mcts_, int batch_size_);
    
    void setup();
    
    std::tuple<Policy, Reward> _convert_n_tie(const torch::Tensor& policy_t,
                                              const torch::Tensor& reward_t);
    
    
    
    // Variables
    MCTS* mcts;
    
    std::string model_path;
    
    std::shared_ptr<torch::jit::script::Module> module;
    
    xt::xtensor<float, 4> input;
    
    std::queue<std::tuple<int, Board>> input_q;
    
    std::thread worker;
    
    std::vector<int> ids;
    
    int batch_size;
    
    bool alive;
    
    std::mutex mut;
    std::condition_variable start_token;
    
    config& conf = config::get();
#ifdef _CUDA
    at::DeviceType device = at::kCUDA;
#else
    at::DeviceType device = at::kCPU;
#endif // _CUDA
};



#endif // evaluator_hpp
