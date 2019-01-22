#include "mcts.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "xtensor-python/pytensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "../blokus/types.hpp"

#include "node.hpp"

void sighandler(int n)
{
    std::stringstream ss;
    ss << "Signal: " << n << " | Thread ID: " << std::this_thread::get_id() << std::endl;
    std::cout << ss.str();
    std::abort();
}

MCTS::MCTS(int iter_budget_, int nthreads_, int batch_size_, float vl_, float c_puct_)
  : iter_budget(iter_budget_),
    nthreads(nthreads_),
    wait_qs(nthreads),
    alive(true),
    evaluated_tokens(nthreads),
    working(false),
    batch_size(batch_size_),
    active_threads(0),
    vl(vl_),
    c_puct(c_puct_),
    alpha(conf.alpha),
    rng(std::random_device{}())
{
    signal(SIGSEGV, sighandler);
    
    for (int i = 0; i < nthreads; i++) {
        std::thread t(&MCTS::_work, this, i);
        threads.push_back(std::move(t));
    }
}


MCTS::~MCTS()
{
    close();
}


void MCTS::close()
{
    alive = false;
    start_token.notify_all();
    
    for (auto& t : threads)
        t.join();
        
    evaluator.close();
}


void MCTS::_work(int idx)
{
    ids.emplace(std::this_thread::get_id(), idx);
    
    while (alive) {
        std::unique_lock<std::mutex> lock(start);
        start_token.wait(lock, [this]{ return working || !alive; });
        lock.unlock();

        if (!alive)
            break;
        
        _begin_simulation(*root);
    }
}


pyPolicy MCTS::search_probs(const pyState& state_, int verbosity_)
{
    verbosity = verbosity_;
    if (verbosity >= 3)
        _log_init();

    
    root = _get_root(_py2cc(state_));
    pyPolicy actions_probs = xt::zeros<float>(conf.action_shape);
    

    // Initialize simulation count to zero.
    count = 0;
    
    
    // Start the threads
    auto start = std::chrono::system_clock::now();
    working = true;
    start_token.notify_all();

    // Wait till done
    std::unique_lock<std::mutex> lock(done);
    done_token.wait(lock, [this]{ return !working; });
    lock.unlock();
    
    // Fill in `action_probs`
    for (Node* child : root->children) {
        
        float prob = (float) child->n / root->n;
        
        // unpack action and use it to index action_probs
        int a,b,c;
        std::tie(a,b,c) = child->action_in;
        
        actions_probs(a,b,c) = prob;
        
    }
    
    
    if (verbosity >= 1) {
        _log_v1(start);
        
        if (verbosity >= 3)
            _log_v3();
    }
    
    return actions_probs;
}



// Private functions

void MCTS::_begin_simulation(Node& root)
{
    count_lock.lock();
    active_threads++;
    while (count < iter_budget) {
        count++;
        if (verbosity >= 2)
            _log_v2();
        count_lock.unlock();
        
        _simulate_once(root);
        
        count_lock.lock();
    }
    if (--active_threads == 0) {
        working = false;
        done_token.notify_one();
    }
    count_lock.unlock();
}

void MCTS::_simulate_once(Node& root)
{
    if (verbosity >= 3) {
        
        auto t0 = std::chrono::system_clock::now();
        Node& leaf = _select(root);
        auto t1 = std::chrono::system_clock::now();
        Reward reward = _eval(leaf);
        auto t2 = std::chrono::system_clock::now();
        _backup(leaf, reward);
        auto t3 = std::chrono::system_clock::now();
        
        count_lock.lock();
        select += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        eval += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        backup += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
        count_lock.unlock();
        
    } else {
        
        Node& leaf = _select(root);
        Reward reward = _eval(leaf);
        _backup(leaf, reward);
        
    }
}

Node& MCTS::_select(Node& root)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _select" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    /*
    `_select` goes on until either the node is terminal, or is a leaf node.
    */
    Node* ptr = &root;
    ptr->lock.lock();
    while (!ptr->terminal && ptr->children.size() > 0) {
        if (ptr->parent)
            ptr->parent->lock.unlock();

        ptr = _choose(ptr->children);
        ptr->n += vl;
        ptr->v -= vl;
        ptr->q = ptr->v / ptr->n;

        ptr->lock.lock();
    }
    ptr->parent->lock.unlock();
    
#ifdef _DEBUG
    std::stringstream sss;
    sss << std::this_thread::get_id() << " will leave _select" << std::endl;
    std::cout << sss.str();
#endif // _DEBUG
    return *ptr;
}

Reward MCTS::_eval(Node& leaf)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _eval" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    if (leaf.terminal) {
#ifdef _DEBUG
        std::stringstream sss;
        sss << std::this_thread::get_id() << " will leave _eval (leaf terminal)" << std::endl;
        std::cout << sss.str();
#endif // _DEBUG
        return leaf.reward;
    } else {

        State state;
        Reward reward;
        bool done;
        
        if (verbosity >= 3) {
            
            auto t0 = std::chrono::system_clock::now();
            std::tie(state, reward, done) = env.step(leaf.parent->state, leaf.action_in);
            auto t1 = std::chrono::system_clock::now();
            count_lock.lock();
            step += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
            step_count++;
            count_lock.unlock();
            
        } else {
            std::tie(state, reward, done) = env.step(leaf.parent->state, leaf.action_in);
        }

        
        if (done) {
            // Add info about the leaf itself
            leaf.add(std::move(state), std::move(reward));
            
        } else {
            const Board& board = std::get<0>(state);
            Policy policy;
            
            int id = ids[std::this_thread::get_id()];
            auto& queue = wait_qs[id];
            
            if (verbosity >= 3) {
                auto t0 = std::chrono::system_clock::now();
                
                consistency_lock.lock();
                evaluator.input_q.push(std::tie(id, board));
                evaluator.start_token.notify_one();
                consistency_lock.unlock();
                
                std::unique_lock<std::mutex> lock(evaluated);
                evaluated_tokens[id].wait(lock, [&queue]{ return !queue.empty(); });
                lock.unlock();
                
                std::tie(policy, reward) = queue.front();
                queue.pop();
                
                auto t1 = std::chrono::system_clock::now();
                count_lock.lock();
                net += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                net_count++;
                count_lock.unlock();
                
                // Add info about the leaf itself
                leaf.add(std::move(state), std::move(reward));
                
                // Add children
                auto t2 = std::chrono::system_clock::now();
                _append_children(leaf, policy);
                auto t3 = std::chrono::system_clock::now();
                count_lock.lock();
                append += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
                count_lock.unlock();
                
            } else {
                consistency_lock.lock();
                evaluator.input_q.push(std::tie(id, board));
                evaluator.start_token.notify_one();
                consistency_lock.unlock();
                
                std::unique_lock<std::mutex> lock(evaluated);
                evaluated_tokens[id].wait(lock, [&queue]{ return !queue.empty(); });
                lock.unlock();
                
                std::tie(policy, reward) = queue.front();
                queue.pop();
                
                // Add info about the leaf itself
                leaf.add(std::move(state), std::move(reward));
                
                // Add children
                _append_children(leaf, policy);
            }
        }
        
#ifdef _DEBUG
        std::stringstream sss;
        sss << std::this_thread::get_id() << " will leave _eval (leaf non-terminal)" << std::endl;
        std::cout << sss.str();
#endif // _DEBUG
        return reward;
    }
}

void MCTS::_backup(Node& leaf, const Reward& result)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _backup" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    Node* node = &leaf;
    while (node->parent) {
        if (node != &leaf) {
            node->lock.lock();
        }

        int player = node->parent->player;
        
        node->n += 1 - vl;
        node->v += result[player] + vl;
        node->q = node->v / node->n;

        node->lock.unlock();
        
        node = node->parent;
    }
    
    // Update root's N
    node->n++;
#ifdef _DEBUG
    std::stringstream sss;
    sss << std::this_thread::get_id() << " will leave _backup" << std::endl;
    std::cout << sss.str();
#endif // _DEBUG
}

Node* MCTS::_choose(const std::vector<Node*>& children)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _choose" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    float max_val = -100;
    std::vector<Node*> max_children;
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
    assert(max_children.size() > 0);
    if (max_children.size() == 1) {
#ifdef _DEBUG
        std::stringstream sss;
        sss << std::this_thread::get_id() << " will leave _choose (single max)" << std::endl;
        std::cout << sss.str();
#endif // _DEBUG
        
        return max_children[0];
        
    } else {
        
        std::uniform_int_distribution<int> dist(0, (int) max_children.size() - 1);
#ifdef _DEBUG
        std::stringstream sss;
        sss << std::this_thread::get_id() << " will leave _choose (multiple max)" << std::endl;
        std::cout << sss.str();
#endif // _DEBUG
        return max_children[dist(rng)];
        
    }
}

Node* MCTS::_get_root(State&& state)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _get_root" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    const Board& board = std::get<0>(state);
    const ID& id = std::get<0>(std::get<1>(state));
    if (tree.find(id) == tree.end()) {
        
        // state doesn't exist in the tree
        tree.clear();
        
        // Create a root - piecewise_construct to assure that no copy/move occurs
        tree.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(id),
            std::forward_as_tuple(this, id, 1, nullptr)
        );
        Node& root = tree.at(id);
        

        Policy policy;
        Reward reward;
        
        int id = 0;
        evaluator.input_q.push(std::tie(id, board));
        evaluator.start_token.notify_one();
        
        auto& queue = wait_qs[id];
        
        std::unique_lock<std::mutex> lock(evaluated);
        evaluated_tokens[id].wait(lock, [&queue]{ return !queue.empty(); });
        lock.unlock();
        
        std::tie(policy, reward) = queue.front();
        queue.pop();
        

        root.add(std::move(state), std::move(reward));
        root.n++;

        _append_children(root, policy);


    } else {
        
        // State does exist in the tree

        // Here we need to leave the current root and its children
        // and erase the rest of the tree. We will accomplish this
        // in two steps. First, we will climb up the tree and remove
        // the parents of the current root as we go up. Once we reach
        // the previous root, we erase the rest of the children, while
        // making sure the current root (and its children) stays intact.

        // If we conduct another search from the same node, it will be
        // found in the tree, but won't have a parent. In such case,
        // skip this entire block and just return the node.
        
        if (tree.at(id).parent) {
            Node* focus = &tree.at(id);
            Node* ptr = focus->parent;
            
            if (!ptr->parent) {
                // ptr is already the root.
                auto it = std::find(ptr->children.begin(), ptr->children.end(), focus);
                ptr->children.erase(it);
            }
            
            // Remove the parents of the current root
            while (ptr->parent) {
                focus = ptr;
                ptr = ptr->parent;
                
                if (!ptr->parent) {
                    // Unregister `focus` from the previous root's children.
                    // Now we can just recursively erase its children and
                    // be sure that the current root is preserved.
                    
                    auto it = std::find(ptr->children.begin(), ptr->children.end(), focus);
                    ptr->children.erase(it);
                }
                // We also need to delete it from the tree as well
                tree.erase(focus->id);
            }
            
            
            // Now recursively erase the previous root's children
            ptr->recursively_erase_children();
            
            // Lastly, erase the previous root itself
            tree.erase(ptr->id);
            
            // Set the current root's parent to nullptr, which
            // signifies that this node is the root.
            tree.at(id).parent = nullptr;
        }
    }
#ifdef _DEBUG
    std::stringstream sss;
    sss << std::this_thread::get_id() << " will leave _get_root" << std::endl;
    std::cout << sss.str();
#endif // _DEBUG
    return &tree.at(id);
}

void MCTS::_append_children(Node& node, Policy& policy)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _append_children" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    const auto& actions = std::get<1>(std::get<1>(node.state))[node.player];
    node.children.reserve(actions.size());
    
    std::vector<float> noise;
    if (!node.parent) {
        // if root, apply dirichlet noise
        noise = _dirichlet((int) actions.size());
    }
    
    int noise_idx = 0;
    for (auto& action : actions) {

        int i, j, k;
        std::tie(i, j, k) = action;
        
        float prior = policy(i, j, k);
        if (!node.parent) {
            prior = 0.75 * prior + 0.25 * noise[noise_idx++];
        }

        ID id = node.id;
        id.push_back(action);

        // Piecewise construct to assure that no copy/move occurs
        create_lock.lock();
        tree.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(id),
            std::forward_as_tuple(this, id, prior, &node)
        );

        node.children.push_back(&tree.at(id));
        create_lock.unlock();
    }
#ifdef _DEBUG
    std::stringstream sss;
    sss << std::this_thread::get_id() << " will leave _append_children" << std::endl;
    std::cout << sss.str();
#endif // _DEBUG
}

std::vector<float> MCTS::_dirichlet(int size, float alpha)
{
#ifdef _DEBUG
    std::stringstream ss;
    ss << std::this_thread::get_id() << " entered _dirichlet" << std::endl;
    std::cout << ss.str();
#endif // _DEBUG
    std::gamma_distribution<float> dist(alpha);
    
    std::vector<float> vec;
    vec.reserve(size);
    
    for (int i = 0; i < size; i++) {
        vec.push_back(dist(rng));
    }
    
    double sum = std::accumulate(vec.begin(), vec.end(), 0.);
    for (float& f : vec)
        f /= sum;
    
#ifdef _DEBUG
    std::stringstream sss;
    sss << std::this_thread::get_id() << " will leave _dirichlet" << std::endl;
    std::cout << sss.str();
#endif // _DEBUG
    return vec;
}


State MCTS::_py2cc(pyState state_)
{

    /*

    To convert a pyState to State, we need to do two things:

    1. Change pyBoard --> Board. This is easily done using xt::adapt().
    2. Change the array type of NewDiagonals from long to unsigned long.
       This should be easy, but since they're so deep down in `Meta`, the code
       for conversion looks quite messy.

    */

    State state;
    Meta& meta = std::get<1>(state);

    // Convert the board
    pyBoard& board_ = std::get<0>(state_);
    Board board = xt::adapt(board_.data(), conf.board_shape);
    std::swap(std::get<0>(state), board);


    // Convert the metadata
    Meta& meta_ = std::get<1>(state_);
    std::swap(std::get<0>(meta), std::get<0>(meta_));
    std::swap(std::get<1>(meta), std::get<1>(meta_));
    std::swap(std::get<3>(meta), std::get<3>(meta_));

    std::vector<NewDiagonals>& new_diags_all_ = std::get<2>(meta_);
    std::vector<NewDiagonals> new_diags_all(new_diags_all_.size());

    for (int i = 0; i < new_diags_all.size(); i++) {
        NewDiagonals& new_diags_ = new_diags_all_[i];
        NewDiagonals new_diags(new_diags_.size());

        for (int j = 0; j < new_diags.size(); j++) {
            Point& pt_ = new_diags_[j];
            new_diags[j] = Point{(unsigned long) pt_[0], (unsigned long) pt_[1]};
        }
        new_diags_all[i] = std::move(new_diags);
    }

    std::swap(std::get<2>(meta), new_diags_all);

    return state;
}

void MCTS::_log_init()
{
    select = std::chrono::milliseconds(0);
    
    eval = std::chrono::milliseconds(0);
    step = std::chrono::milliseconds(0);
    net = std::chrono::milliseconds(0);
    append = std::chrono::milliseconds(0);
    
    step_count = 0;
    net_count = 0;
    
    backup = std::chrono::milliseconds(0);
}

void MCTS::_log_v1(const std::chrono::time_point<std::chrono::system_clock>& start)
{
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::system_clock;
    
    auto duration = duration_cast<milliseconds>(system_clock::now() - start);
    std::cout << "(LOG) threads: " << nthreads << " | iteration: " << count;
    std::cout << " | time(ms): " << duration.count() << std::endl;
}

void MCTS::_log_v2()
{
    std::cout << "* Simulation " << count << " *\r" << std::flush;
}

void MCTS::_log_v3()
{
    // Average of different threads
    float select_f = float(select.count()) / nthreads;
    
    float eval_f = float(eval.count()) / nthreads;
    float   step_f =  float(step.count()) / nthreads;
    float   net_f = float(net.count()) / nthreads;
    float   append_f = float(append.count()) / nthreads;
    
    float backup_f = float(backup.count()) / nthreads;
    
    // To prevent zero division
    if (!count)
        count = 1;
    if (!step_count)
        step_count = 1;
    if (!net_count)
        net_count = 1;
    
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


void MCTS::debug()
{
    Policy policy;
    Reward reward;
    
    Board board = xt::zeros<int>({51,13,13});
    int id = 0;
    
    evaluator.input_q.push(std::tie(id, board));
    evaluator.start_token.notify_one();
    
    auto& queue = wait_qs[id];
    
    std::unique_lock<std::mutex> lock(evaluated);
    evaluated_tokens[id].wait(lock, [&queue]{ return !queue.empty(); });
    lock.unlock();
    
    std::tie(policy, reward) = queue.front();
    queue.pop();
    
    
    std::cout << policy << std::endl;
    std::cout << reward << std::endl;
}
