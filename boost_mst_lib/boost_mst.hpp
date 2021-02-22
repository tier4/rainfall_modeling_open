#include <boost/config.hpp>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <pybind11/pybind11.h>
#include <cmath>

double compMstLength(pybind11::list adj_list, const int node_N);
