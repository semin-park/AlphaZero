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
     * env_       : Environment to search
     * nthreads_  : # of threads to use
     * batch_size_: Maximum batch size for the evaluator
     * conf_      : Configuration object
     */
    MCTS(int nthreads_, int batch_size_, float vl_, float c_puct_, int n_res_, int channels_)
      : nthreads(nthreads_),
        batch_size(batch_size_),
        vl(vl_),
        c_puct(c_puct_),
        n_res(n_res_),
        channels(channels_),
        wait_queues(nthreads),
        wait_tokens(nthreads)
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
                        start_token.wait(lock, [this]{ return working || !alive; }); // escape if working or dead
                        if (!alive)
                            return;
                    }
                    _simulation_loop();
                }
            }, i);
        }
    }

    
    
    /*
     * Destructor joins the threads and closes the evaluator
     */
    ~MCTS()
    {
        alive = false;
        start_token.notify_all();
        
        for (auto& t : threads)
            t.join();
    }
    
    
    
    /*
     * Public API -- returns the policy tensor
     *
     * state_      : Current state of the game
     * iter_budget_: How many iterations to perform
     * verbosity_  : {0,1,2,3}, if greater than 0, this function will print
     *               statistics about the search.
     *               0: Does not print.
     *               1: At the end of the search, print out the number of
     *                  threads and iterations and the duration of the search.
     *               2: During the search, prints out the current iteration number.
     *               3: At the end of the search, print out how much time
     *                  was spent on each stages.
     */
    P search_probs(const S& state_, int iter_budget_, int verbosity_ = 0)
    {
        verbosity = verbosity_;
        if (verbosity >= 3)
            _log_init();
        
        _make_root(state_);
        P actions_probs = torch::zeros(env.get_action_shape());
        
        iter_budget = iter_budget_;
        count = 0;

        working = true;
        start_token.notify_all();
        
        auto start = std::chrono::system_clock::now();
        {
            std::unique_lock<std::mutex> lock(consistency_lock);
            done_token.wait(lock, [this]{ return !working; });
        }

        // Fill in `action_probs`
        for (auto child : root->children) {
            float prob = (float) child->n / root->n;
            
            auto action = child->action_in;

            int i = action[0];
            int j = action[1];

            actions_probs[0][i][j] = prob;
        }
        
        
        if (verbosity >= 1) {
            _log_v1(start);
            if (verbosity >= 3)
                _log_v3();
        }
        
        return actions_probs;
    }
    
    
    void _simulation_loop()
    {
        active_threads++;
        while (true) {
            if (verbosity >= 2) {
                std::unique_lock<std::mutex> lock(consistency_lock);
                _log_v2();
            }
            
            _simulate_once();

            std::unique_lock<std::mutex> lock(consistency_lock);
            if (++count > iter_budget - nthreads) { // count is atomic, but comparison is not, so needs a lock.
                working = false;
                break;
            }
        }
        

        std::unique_lock<std::mutex> lock(consistency_lock);
        if (--active_threads == 0) { // decrement and comparison together are not atomic
            done_token.notify_one();
        }
    }
    
    void _simulate_once()
    {
        if (verbosity >= 3) {
            
            auto t0 = std::chrono::system_clock::now();
            auto leaf = _select();
            auto t1 = std::chrono::system_clock::now();
            R reward = _eval(*leaf);
            auto t2 = std::chrono::system_clock::now();
            _backup(leaf, reward);
            auto t3 = std::chrono::system_clock::now();
            
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                select += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                eval += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                backup += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
            }
            
        } else {
            
            auto leaf = _select();
            R reward = _eval(*leaf);
            _backup(leaf, reward);
            
        }
    }
    
    
    Node<Env>* _select()
    {
        /*
         * _select goes on until either the node is terminal, or is a leaf node.
         * The selected leaf node stays locked.
         *
         * This probably doesn't need a tree level lock.
         */
//        std::shared_lock lock(tree_lock);
        Node<Env>* ptr = root;
        ptr->lock.lock();
        while (!ptr->terminal && ptr->children.size() > 0) {
            if (ptr->parent)
                ptr->parent->lock.unlock();
            
            ptr = _choose(ptr->children);
            ptr->lock.lock();
            ptr->n += vl;
            ptr->v -= vl;
            ptr->q = ptr->v / ptr->n;
        }
        if (ptr->parent)
            ptr->parent->lock.unlock();
        return ptr;
    }
    
    
    R _eval(Node<Env>& leaf)
    {
        if (leaf.terminal)
            return leaf.reward;
        
        S state;
        R reward;
        bool done;
        
        if (verbosity >= 3) {
            
            auto t0 = std::chrono::system_clock::now();
            std::tie(state, reward, done) = env.step(leaf.parent->state, leaf.action_in);
            auto t1 = std::chrono::system_clock::now();
            
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                step += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                step_count++;
            }
            
        } else {
            std::tie(state, reward, done) = env.step(leaf.parent->state, leaf.action_in);
        }
        
        
        if (done) {
            leaf.add(state, reward, true);
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
                evaluator.start_token.notify_one();
                wait_tokens[id].wait(lock, [&queue]{ return !queue.empty(); });
            }
            
            std::tie(policy, reward) = std::move(queue.front());
            queue.pop();
            
            auto t1 = std::chrono::system_clock::now();
            {
                std::unique_lock<std::mutex> lock(consistency_lock);
                net += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                net_count++;
            }
            
            leaf.add(state, reward, false);
            
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
                evaluator.start_token.notify_one();
                wait_tokens[id].wait(lock, [&queue]{ return !queue.empty(); });
            }
            
            std::tie(policy, reward) = std::move(queue.front());
            queue.pop();
            
            leaf.add(state, reward, false);
            
            _append_children(leaf, policy);
        }
        return reward;
    }
    
    void _backup(Node<Env>* leaf, const R& result)
    {
        /*
         * Probably doesn't need tree level synchronization,
         * because the tree is never explicitly used in this
         * function. There is no way that the tree's rebalance
         * could cause problems here.
         *
         * Note: the leaf node has been locked since _select()
         */
        //    std::shared_lock lock(tree_lock);
        Node<Env>* node = leaf;
        while (node->parent) {
            if (node != leaf) {
                node->lock.lock();
            }
            
            int player = node->parent->player;
            
            float value = result[player].item<float>();
            node->n += 1 - vl;
            node->v += value + vl;
            node->q = node->v / node->n;
            
            node->lock.unlock();
            
            node = node->parent;
        }
        node->n++;  // Update root's N
    }
    
    
    
    Node<Env>* _choose(const std::vector<Node<Env>*>& children)
    {
        float max_val = -100;
        std::vector<Node<Env>*> max_children;
        for (auto ptr : children) {
            
            float val = ptr->ucb(c_puct);
            
            if (val > max_val) {
                
                max_val = val;
                max_children.clear();
                max_children.push_back(ptr);
                
            } else if (val == max_val) {
                max_children.push_back(ptr);
            }
        }
        if (max_children.size() == 0)
            throw std::runtime_error("<MCTS::_choose> No max children");
        if (max_children.size() == 1)
            return max_children[0];
        
        std::uniform_int_distribution<int> dist(0, (int) max_children.size() - 1);
        return max_children[dist(rng)];
    }
    
    
    
    void _make_root(const S& state)
    {
        const B& board = env.get_board(state);
        const ID& id = env.get_id(state);
        
        if (tree.find(id) == tree.end()) {
            tree.clear();
            tree.emplace(
                         std::piecewise_construct,
                         std::forward_as_tuple(id),
                         std::forward_as_tuple(id, 1, nullptr)
                         );
            root = &tree.at(id);
            // Run evaluation on the root
            P policy;
            R reward;
            
            int idx = 0;
            auto& queue = wait_queues[idx];
            {
                std::unique_lock<std::mutex> lock(q_lock);
                evaluator.input_q.emplace(idx, board);
                evaluator.start_token.notify_one();
                wait_tokens[idx].wait(lock, [&queue]{ return !queue.empty(); });
            }
            
            std::tie(policy, reward) = std::move(queue.front());
            queue.pop();
            
            root->add(state, reward, false);
            root->n++;
            _append_children(*root, policy);
            
            
        } else if (tree.at(id).parent != nullptr) {
            root = &tree.at(id);
            _prune_tree();
        }
    }

    
    
    void _append_children(Node<Env>& node, const P& policy)
    {
        auto actions = env.possible_actions(node.state, node.player);
        node.children.reserve(actions.size());

        auto policy_a = policy.accessor<float, 2>();
        
        std::vector<float> noise;
        if (!node.parent) {
            // if root, apply dirichlet noise
            noise = dirichlet((int) actions.size());
        }
        
        int noise_idx = 0;
        for (auto& action : actions) {
            
            int i = action[0];
            int j = action[1];

            float prior = policy_a[i][j];
            if (!node.parent) {
                prior = 0.75 * prior + 0.25 * noise[noise_idx++];
            }
            
            ID id = node.id;
            id.push_back(action);
            
            // Piecewise construct to assure that no copy/move occurs
            {
                std::unique_lock<std::shared_mutex> lock(tree_lock);
                tree.emplace(
                             std::piecewise_construct,
                             std::forward_as_tuple(id),
                             std::forward_as_tuple(id, prior, &node)
                             );
            }

            std::shared_lock<std::shared_mutex> lock(tree_lock);
            node.children.push_back(&tree.at(id));
        }
    }
    
    
    
    /*
     * Retains the root, erases everything else
     */
    void _prune_tree()
    {
        if (!root->parent)
            return;
        Node<Env>* focus = root;
        Node<Env>* parent = focus->parent;
        root->parent = nullptr;
        
        auto it = std::find(parent->children.begin(), parent->children.end(), focus);
        parent->children.erase(it);
        
        focus = parent;
        while (focus->parent) {
            focus = focus->parent;
        }
        _erase_descendents(focus);
        tree.erase(focus->id);
    }
    
    /*
     * Recursively erases all decendents of a node.
     * This function is only called by _prune_tree() before
     * executing the search. This function is not thread-safe.
     */
    void _erase_descendents(Node<Env>* top)
    {
        auto& children = top->children;
        for (auto child : children) {
            _erase_descendents(child);
            tree.erase(child->id);
        }
    }

    void load()
    {
        evaluator.setup();
    }
    
    
    /*
     * Attributes
     */

    
    // Nodes need to know
    std::map<ID, Node<Env>> tree; // must be public
    
    // Initialized when MCTS is created
    int nthreads;
    int batch_size;
    float vl;
    float c_puct;
    int n_res;
    int channels;
    
    // Environment
    Env& env = Env::get();
    
    // Runs the actual neural network
    Evaluator<Env>& evaluator = Evaluator<Env>::get(this, batch_size, n_res, channels);


    // How many iterations we've done so far
    std::atomic<int> count{0};
    int iter_budget;
    
    // Multiple threads should know which is the current root
    Node<Env>* root;
    
    // Calculated based on the board size
    float alpha;
    
    // Threads
    std::vector<std::thread> threads;
    std::map<std::thread::id, int> ids;
    std::vector<std::queue<std::tuple<P, R>>> wait_queues;
    std::mutex q_lock; // when you're emplacing to the evaluator queue.
    
    std::atomic<bool> alive {true};
    std::atomic<bool> working {false};
    
    std::atomic<int> active_threads {0};
    
    std::shared_mutex tree_lock; // whenever you modify the structure of the tree
    std::condition_variable start_token;
    std::condition_variable done_token;
    std::vector<std::condition_variable> wait_tokens;
    
    
    
    
    // Random number generator
    std::mt19937 rng = std::mt19937(std::random_device{}());

    // Logging
    int verbosity;
    
    void _log_init()
    {
        select = eval = step = net = append = backup = std::chrono::milliseconds(0);
        step_count = net_count = 0;
    }

    void _log_v1(const std::chrono::time_point<std::chrono::system_clock>& start)
    {
        using std::chrono::duration_cast;
        using std::chrono::milliseconds;
        using std::chrono::system_clock;
        
        auto duration = duration_cast<milliseconds>(system_clock::now() - start);
        std::cout << "(LOG) threads: " << nthreads << " | iteration: " << count;
        std::cout << " | time(ms): " << duration.count() << std::endl;
    }

    void _log_v2()
    {
        std::cout << "* Simulation " << count << " *\r" << std::flush;
    }

    void _log_v3()
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
        
        std::cout << "(Select)     Total: " << std::setw(8) << select_f << " | Avg: " << select_f / count << std::endl;
        std::cout << "(Eval)       Total: " << std::setw(8) << eval_f << " | Avg: " << eval_f / count << std::endl;
        std::cout << "    (Step)   Total: " << std::setw(8) << step_f << " | Avg: " << step_f / step_count << std::endl;
        std::cout << "    (Net)    Total: " << std::setw(8) << net_f << " | Avg: " << net_f / net_count << std::endl;
        std::cout << "    (Append) Total: " << std::setw(8) << append_f << " | Avg: " << append_f / net_count << std::endl;
        std::cout << "(Backup)     Total: " << std::setw(8) << backup_f << " | Avg: " << backup_f / count << std::endl;
        
        std::cout << "Simulation count: " << count << std::endl;
        std::cout << "Step count: " << step_count << std::endl;
        std::cout << "Net count: " << net_count << std::endl;
        std::cout << std::endl;
    }
    
    std::chrono::milliseconds select, eval, step, net, append, backup;
    std::atomic<int> step_count, net_count;
    std::mutex consistency_lock; // whenever you're modifying counting variables, or changing the status of the MCTS

};

#endif /* mcts_hpp */