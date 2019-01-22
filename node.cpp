#include "node.hpp"

#include <cmath>
#include <functional>
#include <iostream>

#include "../blokus/config.hpp"
#include "../blokus/types.hpp"

#include "mcts.hpp"


// This constructor is used when appending empty children to a node.
Node::Node(MCTS* mcts_, const ID& id_, float p_, Node* parent_)
    : tree(&(mcts_->tree)), id(id_), p(p_), parent(parent_), n(0), v(0), q(0), terminal(false)
{
    // When the constructor of Node is called, it only adds the most basic information
    // that is inherited by its parent. This would be an empty tin version of a node.
    // Only when the Node::add function is called is when the node actually materializes.
    
    action_in = id_[id_.size() - 1];
}

// Once a node is expanded, this method is used to put in additional info
void Node::add(State&& state_, Reward&& reward_)
{
    // After env.step(), add the necessary information about the node.
    // This is when this node actually comes to life and stores information about itself.
    state = state_;
    reward = reward_;

    const Board& board = std::get<0>(state);
    player = board(conf.turn, 0, 0);
    terminal = board(conf.done + player, 0, 0);
    
}

float Node::ucb(int c_puct)
{
    return q + c_puct * p * sqrt(parent->n - 1) / (1 + n);
}


void Node::recursively_erase_children()
{
    for (auto child : children) {
        child->recursively_erase_children();
        tree->erase(child->id);
    }
}


// Debugging
std::ostream& operator<< (std::ostream& out, const Node& node)
{
    out << "Node (";
    for (auto a : node.id) {
        out << "(" << std::get<0>(a) << ',' << std::get<1>(a) << ',' << std::get<2>(a) << ")";
    }
    out << ")";
    return out;
}





