#ifndef RANS_CODER_CDF_OPS_H_
#define RANS_CODER_CDF_OPS_H_

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

using CdfList = std::vector<std::vector<uint16_t>>;
using IntList = std::vector<int32_t>;
using PmfArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

constexpr uint32_t CDF_LOOKUP_BITS = 8;
constexpr size_t CDF_LOOKUP_STRIDE = (1u << CDF_LOOKUP_BITS) + 1;

struct CdfTables {
    CdfList cdfs;
    IntList offsets;
    std::vector<uint16_t> inverse_cdfs;
};

CdfTables quantize_pmfs(PmfArray pmfs, IntArray offsets);
void build_inverse_cdfs(CdfTables &tables);

#endif
