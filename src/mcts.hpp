#ifndef mcts_hpp
#define mcts_hpp

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "xtensor-python/pytensor.hpp"
#include "xtensor/xtensor.hpp"

#include "../blokus/Environment.hpp"
#include "../blokus/config.hpp"
#include "../blokus/types.hpp"

#include "evaluator.hpp"
#include "node.hpp"



class MCTS {
friend Evaluator;
public:
    // Constructor
    MCTS(int iter_budget_, int nthreads_, int batch_size_, float vl_ = 3, float c_puct_ = 3);
    ~MCTS();
    void debug();
    

    // Python API
    pyPolicy search_probs(const pyState& state_, int verbosity_ = 0);
    
    
    void close();

    
    // Nodes need to know
    std::map<ID, Node> tree;
    
    int count;

    int iter_budget;

    std::mutex count_lock;
    
private:
    
    void _work(int idx);
    
    void _begin_simulation(Node& root);
    
    void _simulate_once(Node& root);
    
    Node& _select(Node& root);
    
    Reward _eval(Node& leaf);
    
    void _backup(Node& leaf, const Reward& result);
    
    Node* _choose(const std::vector<Node*>& children);
    
    Node* _get_root(State&& state);

    void _append_children(Node& node, Policy& policy);
    
    std::vector<float> _dirichlet(int size, float alpha = 0.05);

    State _py2cc(pyState state_);
    
    
    
    
    
    // Log related (if verbosity == 3)
    void _log_init();
    
    void _log_v1(const std::chrono::time_point<std::chrono::system_clock>& start);
    
    void _log_v2();
    
    void _log_v3();



    // Variables
    Node* root;
    
    config& conf = config::get();
    
    int nthreads;
    std::vector<std::thread> threads;
    std::map<std::thread::id, int> ids;
    std::vector<std::queue<std::tuple<Policy, Reward>>> wait_qs;
    
    std::mutex consistency_lock;
    
    bool alive;
    
    std::mutex start;
    std::condition_variable start_token;
    
    std::mutex done;
    std::condition_variable done_token;
    
    std::mutex evaluated;
    std::vector<std::condition_variable> evaluated_tokens;
    bool working;
    
    int batch_size;
    
    int active_threads;
    
    
    
    Blokus env;
    
    Evaluator& evaluator = Evaluator::get(this, batch_size);
    
    
    
    float vl;
    float c_puct;
    float alpha;
    

    // count_lock is shared between all threads.
    std::mutex create_lock;
    std::mutex eval_lock;
    
    std::mt19937 rng;


    // verbosity
    int verbosity;
    
    
    
    // For logging (if verbosity >= 3)
    std::chrono::milliseconds select;
    
    std::chrono::milliseconds eval;
    std::chrono::milliseconds step;
    std::chrono::milliseconds net;
    std::chrono::milliseconds append;
    
    std::chrono::milliseconds backup;
    
    int step_count;
    int net_count;
    
    std::mutex debug_lock;

};

#endif /* mcts_hpp */
