#ifndef node_hpp
#define node_hpp

#include <map>
#include <mutex>
#include <iostream>
#include <vector>

#include "../blokus/types.hpp"
#include "../blokus/config.hpp"

class MCTS;

struct Node {
    // Constructor
    Node(MCTS* mcts_, const ID& id_, float p_, Node* parent_);

    void recursively_erase_children();
    
    
    // After env.step() details are added
    void add(State&& state_, Reward&& reward_);
    
    float ucb(int c_puct);
    
    
    // Initialized variables
    std::map<ID, Node>* tree;

    ID     id;
    float  p;  // prob as in AGZ

    Node*  parent;
    Action action_in;
    
    int   n;
    float v;
    float q; // v / n
    
    
    // Assigned variables
    State state;
    int player;
    
    bool terminal;
    Reward reward;
    
    
    // MCTS should take care of this
    std::vector<Node*> children;

    // Each Node gets its own lock
    std::mutex lock;
    
    // Share configuration
    config& conf = config::get();
};

std::ostream& operator<< (std::ostream& out, const Node& node);

#endif /* node_hpp */
