#ifndef node_hpp
#define node_hpp

#include <cmath>
#include <mutex>
#include <iostream>
#include <map>

template<class Env>
struct Node {
    using S  = typename Env::state_type;
    using R  = typename Env::reward_type;
    using A  = typename Env::action_type;
    using ID = typename Env::id_type;
    
    /*
     * When Node's constructor is called, it only adds the most basic information
     * that is inherited by its parent. This would be an empty tin version of a node.
     * Only when the Node::add function is called is when the node actually materializes.
     */
    Node(ID id_, float p_, std::shared_ptr<Node> parent_)
      : id(id_), parent(parent_), p(p_) { /* pass */ }
    
    
    /*
     * After env.step(), add() is called to store additional information about the node.
     * This is when this node actually comes to life and stores information about itself.
     */
    void add(const S& state_, int player_, bool done_)
    {
        state = state_;
        player = player_;
        terminal = done_;
    }

    void terminal_add(const S& state_, const R& reward_, int player_, bool done_)
    {
        state = state_;
        reward = reward_;
        player = player_;
        terminal = done_;
    }
    
    /*
     * This calculates the ucb value that is used in the MCTS search.
     */
    float ucb(int c_puct)
    {
        auto par = parent.lock();
        if (par->n < 1)
            throw std::runtime_error("Parent's n is not right");
        if (n == -1)
            throw std::runtime_error("N is -1");
        return q + c_puct * p * sqrt(par->n - 1) / (1 + n);
    }
    
    
    // Initialized variables
    ID id;
    std::weak_ptr<Node> parent;
    S state;
    R reward;
    float p;  // probability of this node being selected

    int   n{0};
    float v{0};
    float q{0};
    
    int player;
    bool terminal {false};
    
    std::map<A, std::shared_ptr<Node>> children;

    /*
     * TODO: Currently, each Node gets its own lock, but
     *       actually you only need nthreads number of locks.
     *       Giving each node its own lock is a waste.
     */
    std::mutex lock;
    
};

template <class Env>
std::shared_ptr<Node<Env>> find(std::shared_ptr<Node<Env>> root, typename Env::id_type key)
{
    if (!root || key.size() < root->id.size()) return nullptr;

    int idx;
    for (idx = 0; idx < (int) root->id.size(); idx++)
        if (root->id[idx] != key[idx]) return nullptr;

    std::shared_ptr<Node<Env>> focus = root;
    for (; idx < (int) key.size(); idx++) {
        const auto& children = focus->children;

        auto it = children.find(key[idx]);
        if (it == children.end()) return nullptr;
        
        focus = it->second;
    }
    return focus;
}

template<class Env>
std::ostream& operator<< (std::ostream& out, const Node<Env>& node)
{
    out << "Node (";
    for (auto a : node.id) {
        out << "(" << std::get<0>(a) << "," << std::get<1>(a) << "," << std::get<2>(a) << ")";
    }
    out << ")";
    return out;
}


#endif /* node_hpp */
