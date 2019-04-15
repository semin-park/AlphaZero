#ifndef evaluator_hpp
#define evaluator_hpp

#include <array>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "netconfig.hpp"
#include "network.hpp"
#include "util.h"

using namespace std::chrono_literals;

template <class Env> class MCTS;

template <class Env>
class Evaluator {
friend class MCTS<Env>;
    
    using S  = typename Env::state_type;
    using ID = typename Env::id_type;
    using A  = typename Env::action_type;
    using B  = typename Env::board_type;
    
    using R  = torch::Tensor;
    using P  = torch::Tensor;
    
public:
    static Evaluator& get(MCTS<Env>* mcts_, int batch_size_, int n_res, int channels)
    {
        static Evaluator singleton(mcts_, batch_size_, n_res, channels);
        return singleton;
    }

private:
    // Evaluator needs to know the mcts instance because it needs to
    // push to the wait queues and notify once evaluated
    Evaluator(MCTS<Env>* mcts_, int batch_size_, int n_res, int channels)
      : mcts(mcts_),
        worker(&Evaluator::run, this),
        batch_size(batch_size_)
    {
        int c_in, c_out, board_size;
        std::tie(c_in, c_out, board_size) = env.get_shape_info();
        NetConfig& netconf = NetConfig::get();
        net = PVNetwork(board_size, netconf.resblocks(), c_in, netconf.channels(), c_out);

        auto shape = env.get_board_shape();
        std::vector<long long> input_shape(shape.begin(), shape.end());
        input_shape.insert(input_shape.begin(), batch_size);
        input = torch::zeros(input_shape);
        ids.reserve(batch_size);
        
        // Neural Network setup
        setup();

        char buffer[200];
        char *cwd = getcwd(buffer, sizeof(buffer));
        std::cout << "* --------------- Evaluator --------------- *" << std::endl
                  << "Model: " << cwd << "/" << model_path << std::endl
                  << "Device: " << (device == torch::kCPU ? "CPU" : "CUDA") << std::endl
                  << "Batch size: " << batch_size << std::endl
                  << "* ----------------------------------------- *" << std::endl;
    }

public:
    void run()
    {
        std::cout << "Evaluator id: " << std::this_thread::get_id() << std::endl;
        while (alive)
        {
            std::unique_lock lock(mut);
            start_token.wait(lock, [this]{ return !input_q.empty() || !alive; });
            if (!alive)
                break;
            
            int size = 0;
            while (size < batch_size) {
                {
                    std::unique_lock lock(mcts->q_lock);
                    if (input_q.empty())
                        break;
                }
                std::tuple<int, B> tup = std::move(input_q.front());
                {
                    std::unique_lock lock(mcts->q_lock);
                    input_q.pop();
                }
                
                
                ids[size] = std::get<0>(tup);
                input.slice(0, size, size + 1) = std::get<1>(tup);
                size++;
            }
            
            const torch::Tensor& X = input.slice(0,0,size);
            torch::Tensor policy, value;
            std::tie(policy, value) = net(X);
            
            auto policy_vec = policy.to(torch::kCPU).chunk(size);
            auto reward_vec = value.to(torch::kCPU).chunk(size);
            
            for (int i = 0; i < size; i++) {
                mcts->wait_queues[ids[i]].emplace(std::move(policy_vec[i][0]), std::move(reward_vec[i][0]));
                mcts->wait_tokens[ids[i]].notify_one();
            }
        }
    }
    
    ~Evaluator()
    {
        {
            std::unique_lock<std::mutex> lock(mut);
            alive = false;
        }
        start_token.notify_one();
        
        worker.join();
    }

    void setup()
    {
        std::string export_path = load_network(net, model_path);

        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            net->to(device);
        }
        
        if (export_path != model_path)
            model_path = export_path;
    }
    
private:
    /*
     * Variables
     */

    Env& env = Env::get();
    
    MCTS<Env>* mcts;
    
    
    bool alive{true};
    
    // neural net related
    std::string model_path;
    PVNetwork net{nullptr};
    torch::Device device{torch::kCPU};

    std::thread worker;
    std::mutex mut;
    std::condition_variable start_token;

    torch::Tensor input;
    std::queue<std::tuple<int, B>> input_q;
    std::vector<int> ids;
    int batch_size;
};



#endif // evaluator_hpp
