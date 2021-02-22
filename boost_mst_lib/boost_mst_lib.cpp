#include <pybind11/pybind11.h>
#include "boost_mst.hpp"

PYBIND11_MODULE(boost_mst_lib, m) {
    m.def("compMstLength", &compMstLength, "Computes the minimum spanning tree length using the C++ Boost library");
}
