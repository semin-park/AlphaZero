#ifndef replay_hpp
#define replay_hpp

#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <random>

#include <torch/torch.h>
#include <zmq.hpp>

template <class Env>
class ReplayBuffer
{
    using T = torch::Tensor;
private:
    Env& env = Env::get();

    int max_size, threshold;
    bool above_threshold{false};

    std::deque<std::tuple<T, T, T>>  buffer; // holds (s, p, z)
    std::deque<std::tuple<T, T>>  temp;      // holds (s, p) -- z is yet unknown

    std::mutex m;
    std::condition_variable cond;
    std::mt19937 rng{std::random_device{}()}; // = std::mt19937(std::random_device{}());

    std::string replay{"replay"};
    std::string state_suffix{".state"};
    std::string policy_suffix{".policy"};
    std::string reward_suffix{".reward"};

    enum { GEN, TRN };
    zmq::context_t context;
    zmq::socket_t socket;

public:
    // Note: 'train_threshold' is only used by the Trainer.
    ReplayBuffer(int ID, const std::string& IP, const std::string& port, int max = 1, int train_threshold = 0)
      : max_size(max), threshold(train_threshold), context(1), socket(context, ID ? ZMQ_PULL : ZMQ_PUSH)
    {
        if (max < threshold)
            throw std::runtime_error(
                "Max buffer size should be greater than the training threshold.");

        std::string path = "tcp://" + IP + ":" + port;
        std::cout << "Path: " << path << std::endl;
        if (ID == GEN)
            socket.bind(path);
        else
            socket.connect(path);
    }

    void temporary_append(const T& state, const T& policy)
    {
        // Generator API
        temp.emplace_back(state, policy);
    }

    void send_reward(const T& reward)
    {
        // Generator API
        // if (reward[0].item<float>() == 0) {
        //     static std::uniform_real_distribution dist(0, 1);
        //     if (dist(rng) < 0.9)
        //         return;
        // }
        int n = temp.size();
        T state, policy;
        int state_size, policy_size, reward_size;

        for (int i = 0; i < n; i++) {
            std::tie(state, policy) = std::move(temp.front());
            temp.pop_front();

            if (i == 0) {
                state_size = state.numel() * sizeof(uint8_t);
                policy_size = policy.numel() * sizeof(float);
                reward_size = reward.numel() * sizeof(float);
            }

            zmq::message_t msg_s(state_size);
            zmq::message_t msg_p(policy_size);
            zmq::message_t msg_r(reward_size);

            memcpy(msg_s.data(), state.data<uint8_t>(), state_size);
            memcpy(msg_p.data(), policy.data<float>(), policy_size);
            memcpy(msg_r.data(), reward.data<float>(), reward_size);

            socket.send(msg_s, ZMQ_SNDMORE);
            socket.send(msg_p, ZMQ_SNDMORE);
            socket.send(msg_r, 0);

            // std::cout << "Sent everything." << std::endl;
            // std::cout << "State:\n" << state << std::endl;
            // std::cout << "Policy:\n" << policy << std::endl;
            // std::cout << "Reward:\n" << reward << std::endl;
        }
    }



    // Trainer API

    void receive()
    {
        auto b_sh = env.get_board_shape();
        auto a_sh = env.get_action_shape();

        T state = torch::empty({b_sh[0], b_sh[1], b_sh[2]}, torch::TensorOptions().dtype(torch::kUInt8));
        T policy = torch::empty({a_sh[0], a_sh[1], a_sh[2]});
        T reward = torch::empty({2});

        int more;
        size_t more_size = sizeof(int);
        zmq::message_t msg_s, msg_p, msg_r;
        
        socket.recv(&msg_s);
        socket.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        if (!more)
            throw std::runtime_error("Not enough messages!");

        socket.recv(&msg_p);
        socket.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        if (!more)
            throw std::runtime_error("Not enough messages!");

        socket.recv(&msg_r);
        socket.getsockopt(ZMQ_RCVMORE, &more, &more_size);
        if (more)
            throw std::runtime_error("Too many messages!");

        memcpy(state.data<uint8_t>(), msg_s.data(), state.numel() * sizeof(uint8_t));
        memcpy(policy.data<float>(), msg_p.data(), policy.numel() * sizeof(float));
        memcpy(reward.data<float>(), msg_r.data(), reward.numel() * sizeof(float));

        // std::cout << "Got everything." << std::endl;
        // std::cout << "State:\n" << state << std::endl;
        // std::cout << "Policy:\n" << policy << std::endl;
        // std::cout << "Reward:\n" << reward << std::endl;

        if (buffer.size() >= max_size)
            buffer.pop_front();

        {
            std::unique_lock lock(m);
            buffer.emplace_back(std::move(state), std::move(policy), std::move(reward));
            if (buffer.size() == threshold) {
                above_threshold = true;
                cond.notify_one();
            }
        }
    }


    void load(std::string dir)
    {
        std::cout << "Loading buffer." << std::endl;
        if (dir[dir.size() - 1] != '/')
            dir += "/";

        std::string state_path = dir + replay + state_suffix;
        std::string policy_path = dir + replay + policy_suffix;
        std::string reward_path = dir + replay + reward_suffix;

        std::ifstream f(state_path.c_str());
        if (!f.good()) {
            std::cout << "Path doesn't exist. Skipping load." << std::endl;
            return;
        }

        T state, policy, reward;

        torch::load(state, state_path);
        torch::load(policy, policy_path);
        torch::load(reward, reward_path);

        int n = state.sizes()[0];

        auto state_vec = state.chunk(n);
        auto policy_vec = policy.chunk(n);
        auto reward_vec = reward.chunk(n);

        std::unique_lock lock(m);
        for (int i = 0; i < n; i++) {
            if (buffer.size() == threshold) {
                above_threshold = true;
                cond.notify_one();
            }

            buffer.emplace_back(
                std::move(state_vec[i][0]),
                std::move(policy_vec[i][0]),
                std::move(reward_vec[i][0])
            );

            if (buffer.size() >= max_size)
                buffer.pop_front();
        }
        std::cout << "Buffer loaded. Size: " << buffer.size() << std::endl;
    }

    std::tuple<T,T,T> get_batch(int size)
    {
        // buffer.size() must be greater than threshold
        std::unique_lock lock(m);

        if (threshold < size)
            throw std::runtime_error("Size must be smaller than the threshold.");

        if (buffer.size() < threshold) {
            std::cout << "Buffer size is smaller than the threshold. Will wait." << std::endl;
            cond.wait(lock, [this] { return above_threshold; });
        }
            

        auto b_sh = env.get_board_shape();
        auto a_sh = env.get_action_shape();

        T state = torch::empty({size, b_sh[0], b_sh[1], b_sh[2]});
        T policy = torch::empty({size, a_sh[0], a_sh[1], a_sh[2]});
        T reward = torch::empty({size, 2});

        std::uniform_int_distribution<int> dist(0, (int) buffer.size() - 1);
        for (int i = 0; i < size; i++) {
            auto tup = buffer[dist(rng)];
            state.slice(0,i,i+1) = std::get<0>(tup);
            policy.slice(0,i,i+1) = std::get<1>(tup);
            reward.slice(0,i,i+1) = std::get<2>(tup);
        }
        return {state, policy, reward};
    }

    void save(std::string dir)
    {
        if (dir[dir.size() - 1] != '/')
            dir += "/";

        std::string state_path = dir + replay + state_suffix;
        std::string policy_path = dir + replay + policy_suffix;
        std::string reward_path = dir + replay + reward_suffix;

        int n = buffer.size();

        auto b_sh = env.get_board_shape();
        auto a_sh = env.get_action_shape();

        T state = torch::empty({n, b_sh[0], b_sh[1], b_sh[2]});
        T policy = torch::empty({n, a_sh[0], a_sh[1], a_sh[2]});
        T reward = torch::empty({n, 2});

        {
            std::unique_lock lock(m);
            for (int i = 0; i < n; i++) {
                state.slice(0,i,i+1) = std::get<0>(buffer[i]);
                policy.slice(0,i,i+1) = std::get<1>(buffer[i]);
                reward.slice(0,i,i+1) = std::get<2>(buffer[i]);
            }
        }
        torch::save(state, state_path);
        torch::save(policy, policy_path);
        torch::save(reward, reward_path);
    }

    int size()
    {
        return buffer.size();
    }
};

#endif /* replay_hpp */