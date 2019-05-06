#ifndef mcts_hpp
#define mcts_hpp

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>

#include "evaluator.hpp"
#include "node.hpp"
#include "util.h"

template<class Env>
class MCTS {
friend class Evaluator<Env>;
public:
    // types
    using S  = typename Env::state_type;
    using ID = typename Env::id_type;
    using A  = typename Env::action_type;
    
    using B  = torch::Tensor;
    using R  = torch::Tensor;
    using P  = torch::Tensor;
    
    
    /*
     * Constructor for the MCTS.
     * This initializes given number of workers
     *
     * nthreads_   : # of threads to use
     * batch_size_ : Maximum batch size for the evaluator
     * vl_         : Virtual loss
     * c_puct_     : Amount of exploration (used in calculating nodes' ucb values)
     */
    MCTS(int nthreads_, int batch_size_, float vl_, float c_puct_)
      : nthreads(nthreads_),
        batch_size(batch_size_),
        vl(vl_),
        c_puct(c_puct_),
        wait_queues(nthreads),
        wait_conds(nthreads)
    {
        signal(SIGSEGV, sighandler);

        alpha = 10. / (env.get_board_size() * env.get_board_size());

        std::cout << "MCTS id: " << std::this_thread::get_id() << std::endl;
        for (int i = 0; i < nthreads; i++) {
            threads.emplace_back([this] (int idx) {
                {
                    std::unique_lock<std::mutex> lock(consistency_lock);
                    ids.emplace(std::this_thread::get_id(), idx);
                    std::cout << "Thread " << idx << " id: " << std::this_thread::get_id() << std::endl;
                }
                
                while (alive) {
                    {
                        std::unique_lock<std::mutex> lock(consistency_lock);
                        start_cond.wait(lock, [this]{ return working || !alive; }); // escape if working or dead
                        if (!alive)
                            return;
                    }
                    __simulation_loop();
                }
            }, i);
        }
    }

    
    
    /*
     * Join threads and close the evaluator
     */
    ~MCTS()
    {
        alive = false;
        start_cond.notify_all();
        
        for (auto& t : threads)
            t.join();
    }
    
    
    
    /*
     * Public API -- returns the policy tensor
     *
     * state_       : Current state of the game
     * iter_budget_ : How many iterations to perform
     * verbosity_   : {0,1,2,3}, if greater than 0, this function will print
     *                statistics about the search.
     *                0: Does not print.
     *                1: At the end of the search, print out the number of
     *                   threads and iterations and the duration of the search.
     *                2: During the search, prints out the current iteration.
     *                3: At the end of the search, prints out how much time
     *                   was spent on each step.
     */
    P search_probs(const S& state_, int iter_budget_, int verbosity_ = 0)
    {
        verbosity = verbosity_;
        if (verbosity >= 3)
            __log_init();

        auto start = std::chrono::system_clock::now();

        evaluator.reset_stat();

        if (verbosity >= 3) {
            auto t0 = std::chrono::system_clock::now();
            __make_root(state_);
            auto t1 = std::chrono::system_clock::now();
            create = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        } else {
            __make_root(state_);
        }
        
        P actions_probs = torch::zeros(env.get_action_shape());
        
        iter_budget = iter_budget_;
        count = 0;

        working = true;
        start_cond.notify_all();
        {
            std::unique_lock<std::mutex> lock(consistency_lock);
            done_cond.wait(lock, [this]{ return !working; });
        }
        
        auto children = root->children;
        for (auto it = children.begin(); it != children.end(); it++) {
            auto action = it->first;
            auto child = it->second;
            float prob = (float) child->n / root->n;

            int i = action[0];
            int j = action[1];

            actions_probs[0][i][j] = prob;
        }
        
        auto evaluator_stat = evaluator.retrieve_stat();
        if (verbosity >= 1) {
            __log_v1(start, evaluator_stat);
            if (verbosity >= 3)
                __log_v3();
        }
        return actions_probs;
    }
    
    
    void __simulation_loop()
    {
        active_threads++;
        while (true) {
            if (verbosity >= 2) {
                std::unique_lock<std::mutex> lock(consistency_lock);
                __log_v2();
            }
            
            __simulate_once();

            std::unique_lock<std::mutex> lock(consistency_lock);
            if (++count > iter_budget - nthreads) { // count is atomic, but comparison is not, so needs a lock.
                working = false;
                break;
            }
        }
        

        std::unique_lock<std::mutex> lock(consistency_lock);
        if (--active_threads == 0) { // decrement and comparison together are not atomic
            done_cond.notify_one();
        }
    }
    
    void __simulate_once()
    {
        std::shared_ptr<Node<Env>> leaf;
        R reward;
        if (verbosity >= 3) {
            
            auto t0 = std::chrono::system_clock::now();
            auto t1 = std::chrono::system_clock::now();
            std::tie(leaf, reward) = __select_and_eval();
            auto t2 = std::chrono::system_clock::now();
            __backup(leaf, reward);
            auto t3 = std::chrono::system_clock::now();
            
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                select += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                eval += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                backup += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
            }
            
        } else {
            std::tie(leaf, reward) = __select_and_eval();
            __backup(leaf, reward);
            
        }
    }
    
    
    std::tuple<std::shared_ptr<Node<Env>>, R> __select_and_eval()
    {
        auto focus = root;
        typename decltype(focus->children)::iterator it;

        focus->lock.lock();
        while (!focus->terminal && focus->children.size() > 0) {
            auto parent = focus->parent.lock();
            if (parent)
                parent->lock.unlock();
            
            it = __choose(focus->children);
            focus = it->second;
            focus->lock.lock();
            focus->n += vl;
            focus->v -= vl;
            focus->q = focus->v / focus->n;
        }
        auto parent = focus->parent.lock();
        if (parent)
            parent->lock.unlock();
        if (focus->terminal)
            return std::make_tuple(focus, focus->reward);

        R reward = _eval(focus, it->first);
        return std::make_tuple(focus, reward);
    }
    
    
    R _eval(std::shared_ptr<Node<Env>>& leaf, const A& action)
    {
        const auto& parent = leaf->parent.lock();
        S state;
        R reward;
        int player;
        bool done;
        
        if (verbosity >= 3) {
            
            auto t0 = std::chrono::system_clock::now();
            std::tie(state, reward, done) = env.step(parent->state, action);
            auto t1 = std::chrono::system_clock::now();
            
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                step += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                step_count++;
            }
            
        } else {
            std::tie(state, reward, done) = env.step(parent->state, action);
        }

        player = env.get_player(state);
        
        if (done) {
            leaf->terminal_add(state, reward, player, true);
            return reward;
        }
        
        const B& board = env.get_board(state);
        P policy;
        int id = ids[std::this_thread::get_id()];
        auto& queue = wait_queues[id];
        
        if (verbosity >= 3) {
            auto t0 = std::chrono::system_clock::now();
            {
                std::unique_lock<std::mutex> lock(q_lock);
                evaluator.input_q.emplace(id, board);
                evaluator.start_cond.notify_one();
                wait_conds[id].wait(lock, [&queue]{ return !queue.empty(); });
            }
            
            std::tie(policy, reward) = std::move(queue.front());
            queue.pop();
            
            auto t1 = std::chrono::system_clock::now();
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                net += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                net_count++;
            }
            
            leaf->add(state, player, false);
            
            auto t2 = std::chrono::system_clock::now();
            _append_children(leaf, policy);
            auto t3 = std::chrono::system_clock::now();
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                append += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
            }
            
        } else {
            {
                std::unique_lock<std::mutex> lock(q_lock);
                evaluator.input_q.emplace(id, board);
                evaluator.start_cond.notify_one();
                wait_conds[id].wait(lock, [&queue]{ return !queue.empty(); });
            }
            
            std::tie(policy, reward) = std::move(queue.front());
            queue.pop();
            
            leaf->add(state, player, false);
            
            _append_children(leaf, policy);
        }
        return reward;
    }
    
    void __backup(std::shared_ptr<Node<Env>>& leaf, const R& result)
    {
        /*
         * Note: the leaf node is locked since __select()
         */
        std::shared_ptr<Node<Env>> node = leaf;
        while (auto parent = node->parent.lock()) {
            if (node != leaf) {
                node->lock.lock();
            }
            
            int player = parent->player;
            
            float value = result[player].item<float>();
            node->n += 1 - vl;
            node->v += value + vl;
            node->q = node->v / node->n;
            
            node->lock.unlock();
            
            node = parent;
        }
        node->n++;  // Update root's N
    }
    
    
    
    auto __choose(std::map<A, std::shared_ptr<Node<Env>>>& children)
        -> typename std::remove_reference<decltype(children)>::type::iterator
    {
        using result_type = typename std::remove_reference<decltype(children)>::type::iterator;

        float max_val = -100;
        result_type it;
        std::vector<result_type> max_children;

        for (it = children.begin(); it != children.end(); it++) {
            auto node = it->second;
            
            float val = node->ucb(c_puct);
            
            if (val > max_val) {
                
                max_val = val;
                max_children.clear();
                max_children.push_back(it);
                
            } else if (val == max_val) {
                max_children.push_back(it);
            }
        }
        if (max_children.size() == 0)
            throw std::runtime_error("<MCTS::__choose> No max children");
        if (max_children.size() == 1)
            return max_children[0];
        
        std::uniform_int_distribution<int> dist(0, (int) max_children.size() - 1);
        return max_children[dist(rng)];
    }
    
    
    
    void __make_root(const S& state)
    {
        const B& board = env.get_board(state);
        const ID& id = env.get_id(state);
        int player = env.get_player(state);
        
        auto target = find(root, id);
        if (target == nullptr) {
            std::cout << "Creating a new root" << std::endl;
            root = std::make_shared<Node<Env>>(id, 1, nullptr);
            
            // Run evaluation on the root
            P policy;
            R reward;
            
            int idx = 0;
            auto& queue = wait_queues[idx];
            {
                std::unique_lock<std::mutex> lock(q_lock);
                evaluator.input_q.emplace(idx, board);
                evaluator.start_cond.notify_one();
                wait_conds[idx].wait(lock, [&queue]{ return !queue.empty(); });
            }
            
            std::tie(policy, reward) = std::move(queue.front());
            queue.pop();
            
            root->add(state, player, false);
            root->n++;
            _append_children(root, policy);
            
            
        } else if (target->parent.lock()) {
            std::cout << "Reusing root" << std::endl;
            root = target;
            std::cout << "Pruned tree" << std::endl;
        }
    }

    
    
    void _append_children(std::shared_ptr<Node<Env>>& node, const P& policy)
    {
        auto actions = env.possible_actions(node->state, node->player);
        auto& children = node->children;

        auto policy_a = policy.accessor<float, 2>();
        std::vector<float> noise;
        auto parent = node->parent.lock();
        if (!parent) {
            noise = dirichlet((int) actions.size());
        }
        int noise_idx = 0;
        for (const auto& action : actions) {

            int i = action[0];
            int j = action[1];

            float prior = policy_a[i][j];
            if (!parent) {
                prior = 0.75 * prior + 0.25 * noise[noise_idx++];
            }

            ID id = node->id;
            id.push_back(action);

            // Piecewise construct to assure that no copy/move occurs
            auto child = std::make_shared<Node<Env>>(id, prior, node);
            children.emplace(action, child);
        }
    }
    
    void load()
    {
        evaluator.setup();
    }

    void clear()
    {
        root.reset();
    }
    
    
    /*
     * Attributes
     */
    
    int nthreads;
    int batch_size;
    float vl;
    float c_puct;
    
    // Environment
    Env& env = Env::get();
    
    // Runs the actual neural network
    Evaluator<Env>& evaluator = Evaluator<Env>::get(this, batch_size);


    // How many iterations we've done so far
    std::atomic<int> count{0};
    int iter_budget;
    
    // Multiple threads should know which is the current root
    std::shared_ptr<Node<Env>> root{nullptr};
    
    // Calculated based on the board size
    float alpha;
    
    // Threads
    std::vector<std::thread> threads;
    std::map<std::thread::id, int> ids;
    std::vector<std::queue<std::tuple<P, R>>> wait_queues;
    std::mutex q_lock; // when you're emplacing to the evaluator queue.
    
    std::atomic<bool> alive {true}, working {false};
    
    std::atomic<int> active_threads {0};
    
    std::condition_variable start_cond, done_cond;
    std::vector<std::condition_variable> wait_conds;
    
    
    // Random number generator
    std::mt19937 rng = std::mt19937(std::random_device{}());

    // Logging
    int verbosity;
    
    void __log_init()
    {
        select = eval = step = net = append = backup = std::chrono::milliseconds(0);
        step_count = net_count = 0;
    }

    void __log_v1(const std::chrono::time_point<std::chrono::system_clock>& start,
        const std::tuple<float, int>& evaluator_stat)
    {
        using std::chrono::duration_cast;
        using std::chrono::milliseconds;
        using std::chrono::system_clock;

        float avg_size; int nn_count;
        std::tie(avg_size, nn_count) = evaluator_stat;
        
        auto duration = duration_cast<milliseconds>(system_clock::now() - start);
        std::cout << "(LOG) threads: " << nthreads
                  << " | iteration: " << count
                  << " | average batch size: " << avg_size
                  << " | NN count: " << nn_count
                  << " | time(ms): " << duration.count()
                  << std::endl;
    }

    void __log_v2()
    {
        std::cout << "* Simulation " << count << " *\r" << std::flush;
    }

    void __log_v3()
    {
        // Average of different threads
        float select_f = float(select.count()) / nthreads;
        
        float eval_f   = float(eval.count())   / nthreads;
        float step_f   = float(step.count())   / nthreads;
        float net_f    = float(net.count())    / nthreads;
        float append_f = float(append.count()) / nthreads;
        
        float backup_f = float(backup.count()) / nthreads;
        
        // To prevent zero division
        if (count == 0) count++;
        if (step_count == 0) step_count++;
        if (net_count == 0) net_count++;
        
        std::cout << "(Root Prune) Total: " << std::setw(8) << create.count() << std::endl;
        std::cout << "(Select)     Total: " << std::setw(8) << select_f << " | Avg: " << select_f / count     << std::endl;
        std::cout << "(Eval)       Total: " << std::setw(8) << eval_f   << " | Avg: " << eval_f / count       << std::endl;
        std::cout << "    (Step)   Total: " << std::setw(8) << step_f   << " | Avg: " << step_f / step_count  << std::endl;
        std::cout << "    (Net)    Total: " << std::setw(8) << net_f    << " | Avg: " << net_f / net_count    << std::endl;
        std::cout << "    (Append) Total: " << std::setw(8) << append_f << " | Avg: " << append_f / net_count << std::endl;
        std::cout << "(Backup)     Total: " << std::setw(8) << backup_f << " | Avg: " << backup_f / count     << std::endl;
        
        std::cout << "Simulation count: " << count << std::endl;
        std::cout << "Step count: " << step_count << std::endl;
        std::cout << "Net count: " << net_count << std::endl;
        std::cout << std::endl;
    }
    
    std::chrono::milliseconds create, select, eval, step, net, append, backup;
    std::atomic<int> step_count, net_count;
    std::mutex consistency_lock; // whenever you're modifying counting variables, or changing the status of the MCTS

};

#endif /* mcts_hpp */