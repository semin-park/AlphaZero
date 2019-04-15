# alpha_mcts
Implementation of MCTS in AlphaGo Zero



`class MCTS`

- `MCTS::MCTS(int nthreads_, int batch_size_, config conf);`
  - Constructs the MCTS class and creates a bunch of threads to use for the tree search. `config conf` includes the configuration for the search.
- `MCTS::set_env(Environment env);`
  - Sets the environment to be used for the tree search.
- `MCTS::search_probs(const State& state, int iter_budget, int verbosity_ = 0);`
  - Conducts `iter_budget` number of tree search from the given `state`.



`class Evaluator`

- 