#include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
#include "pybind11/stl.h"                 // map, vector, etc
#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "../blokus/Environment.hpp"

#include "mcts.hpp"

PYBIND11_MODULE(mcts, m)
{
    xt::import_numpy();
    m.doc() = "MCTS";
    
    pybind11::class_<MCTS>(m, "MCTS")
    .def(pybind11::init<int, int, int, float, float>(),
        pybind11::arg("iter_budget"),
        pybind11::arg("nthreads"),
        pybind11::arg("batch_size"),
        pybind11::arg("virtual_loss") = 3,
        pybind11::arg("c_puct") = 3
    )
    .def("search_probs", &MCTS::search_probs, pybind11::arg("state"), pybind11::arg("verbosity") = 0)
    ;
}
