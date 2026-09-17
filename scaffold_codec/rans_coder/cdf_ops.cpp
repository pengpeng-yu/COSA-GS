#include "cdf_ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

constexpr uint32_t CDF_TOTAL = 1u << 16;

std::vector<uint16_t> quantize_pmf(double *pmf, size_t size,
                                   int32_t &offset) {
    for (size_t i = 0; i < size; ++i) {
        if (pmf[i] < 0 || !std::isfinite(pmf[i]))
            throw std::runtime_error("PMF values must be finite and nonnegative.");
    }

    const double pmf_total = std::accumulate(pmf, pmf + size, 0.0);
    const double total = std::max(pmf_total, 1.0);
    std::partial_sum(pmf, pmf + size, pmf);

    std::vector<uint32_t> full(size + 2, 0);
    for (size_t i = 0; i < size; ++i)
        full[i + 1] = static_cast<uint32_t>(
            std::round(CDF_TOTAL * pmf[i] / total));
    full.back() = CDF_TOTAL;

    size_t first = size;
    size_t last = 0;
    for (size_t i = 0; i < size; ++i) {
        if (full[i + 1] != full[i]) {
            if (first == size)
                first = i;
            last = i;
        }
    }

    if (first == size) {
        offset += static_cast<int32_t>(size);
        return {};
    }

    offset += static_cast<int32_t>(first);
    const size_t regular_symbols = last - first + 1;
    std::vector<uint32_t> frequencies(regular_symbols + 1);
    for (size_t i = 0; i < regular_symbols; ++i)
        frequencies[i] = full[first + i + 1] - full[first + i];
    frequencies.back() = CDF_TOTAL - full[last + 1];

    uint32_t missing = 0;
    std::vector<size_t> donors;
    donors.reserve(frequencies.size());
    for (size_t i = 0; i < frequencies.size(); ++i) {
        if (frequencies[i] == 0) {
            frequencies[i] = 1;
            ++missing;
        } else if (frequencies[i] > 1) {
            donors.push_back(i);
        }
    }
    if (missing != 0) {
        std::sort(donors.begin(), donors.end(),
                  [&](size_t left, size_t right) {
            return frequencies[left] > frequencies[right];
        });
        for (size_t donor : donors) {
            const uint32_t taken = std::min(
                frequencies[donor] - 1, missing);
            frequencies[donor] -= taken;
            missing -= taken;
            if (missing == 0)
                break;
        }
    }
    if (missing != 0)
        throw std::runtime_error("CDF has insufficient frequency mass.");

    std::vector<uint16_t> cdf(regular_symbols);
    uint32_t cumulative = 0;
    for (size_t i = 0; i < regular_symbols; ++i) {
        cumulative += frequencies[i];
        cdf[i] = static_cast<uint16_t>(cumulative);
    }
    return cdf;
}

}  // namespace

CdfTables quantize_pmfs(PmfArray pmfs, IntArray offsets) {
    const py::buffer_info pmf_info = pmfs.request();
    const py::buffer_info offset_info = offsets.request();
    if (pmf_info.ndim != 2 || offset_info.ndim != 1
        || pmf_info.shape[0] != offset_info.shape[0])
        throw std::runtime_error("Expected PMFs [N,M] and offsets [N].");

    const size_t rows = static_cast<size_t>(pmf_info.shape[0]);
    const size_t columns = static_cast<size_t>(pmf_info.shape[1]);
    auto *offset_data = static_cast<int32_t *>(offset_info.ptr);
    CdfList cdfs(rows);

    py::gil_scoped_release release;
    for (size_t row = 0; row < rows; ++row) {
        auto *pmf = reinterpret_cast<double *>(
            static_cast<uint8_t *>(pmf_info.ptr) + row * pmf_info.strides[0]);
        cdfs[row] = quantize_pmf(pmf, columns, offset_data[row]);
    }
    return {
        std::move(cdfs),
        IntList(offset_data, offset_data + rows),
        {},
    };
}

void build_inverse_cdfs(CdfTables &tables) {
    py::gil_scoped_release release;
    tables.inverse_cdfs.resize(tables.cdfs.size() * CDF_LOOKUP_STRIDE);
    for (size_t index = 0; index < tables.cdfs.size(); ++index) {
        const auto &cdf = tables.cdfs[index];
        auto *inverse = tables.inverse_cdfs.data() + index * CDF_LOOKUP_STRIDE;
        size_t symbol = 0;
        for (size_t bucket = 0; bucket < CDF_LOOKUP_STRIDE; ++bucket) {
            const uint32_t cumulative = bucket * (CDF_TOTAL >> CDF_LOOKUP_BITS);
            while (symbol < cdf.size() && cdf[symbol] <= cumulative)
                ++symbol;
            inverse[bucket] = static_cast<uint16_t>(symbol);
        }
    }
}
