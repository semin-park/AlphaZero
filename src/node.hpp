#ifndef node_hpp
#define node_hpp

#include <cmath>
#include <mutex>
#include <iostream>
#include <vector>

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
    Node(const ID& id_, float p_, Node* parent_)
      : parent(parent_), id(id_), p(p_)
    {
        action_in = id_.at(id_.size() - 1);
    }
    
    
    /*
     * After env.step(), add() is called to store additional information about the node.
     * This is when this node actually comes to life and stores information about itself.
     */
    void add(const S& state_, const R& reward_, bool done_)
    {
        state = state_;
        reward = reward_;
        terminal = done_;
        
        player = env.get_player(state_);
    }
    
    /*
     * This calculates the ucb value that is used in the MCTS search.
     */
    float ucb(int c_puct)
    {
        if (parent->n - 1 < 0)
            throw std::runtime_error("Parent's n is not right");
        if (n == -1)
            throw std::runtime_error("N is -1");
        return q + c_puct * p * sqrt(parent->n - 1) / (1 + n);
    }
    
    
    // Initialized variables
    Env& env = Env::get();

    Node* parent;
    ID   id;
    S state;
    A action_in;
    R reward;
    float p;  // probability of this node being selected

    int   n{0};
    float v{0};
    float q{0}; // q = v / n
    
    int player;
    
    bool terminal{false};
    
    std::vector<Node<Env>*> children;

    /*
     * TODO: Currently, each Node gets its own lock, but
     *       actually you only need nthreads number of locks.
     *       Giving each node its own lock is a waste.
     */
    std::mutex lock;
    
};

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
