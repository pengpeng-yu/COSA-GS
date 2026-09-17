#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "cdf_ops.h"
#include "rans_core.h"

namespace py = pybind11;

namespace {

constexpr uint32_t PRECISION = 16;
constexpr uint32_t CDF_TOTAL = 1u << PRECISION;
constexpr py::ssize_t BUFFER_CHECK_INTERVAL = 4096;
using SymbolArray = py::array_t<int16_t,
    py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<uint16_t,
    py::array::c_style | py::array::forcecast>;
using CdfArray = py::array_t<uint16_t,
    py::array::c_style | py::array::forcecast>;

class RansEncoder {
public:
    RansEncoder() : buffer_(new uint8_t[capacity_]) { reset(); }

    void encode_indexed(SymbolArray symbols, IndexArray indexes,
                        const CdfTables &tables) {
        const py::buffer_info symbol_info = symbols.request();
        const py::buffer_info index_info = indexes.request();
        if (symbol_info.size != index_info.size)
            throw std::runtime_error("Symbol and index counts differ.");

        const auto *symbol_data = static_cast<const int16_t *>(symbol_info.ptr);
        const auto *index_data = static_cast<const uint16_t *>(index_info.ptr);
        py::gil_scoped_release release;
        for (py::ssize_t end = symbol_info.size; end > 0;) {
            const py::ssize_t begin = end - std::min(end, BUFFER_CHECK_INTERVAL);
            ensure_capacity(static_cast<size_t>(end - begin) * 12 + 16);
            for (py::ssize_t position = end; position-- > begin;) {
                const uint16_t table_index = index_data[position];
                encode_indexed_symbol(
                    static_cast<int32_t>(symbol_data[position]),
                    table_index, tables);
            }
            end = begin;
        }
    }

    void encode_categorical(CdfArray intervals) {
        const py::buffer_info info = intervals.request();
        if (info.ndim != 2 || info.shape[1] != 2)
            throw std::runtime_error("Expected rANS intervals [N,2].");

        const auto *data = static_cast<const uint16_t *>(info.ptr);
        py::gil_scoped_release release;
        for (py::ssize_t end = info.shape[0]; end > 0;) {
            const py::ssize_t begin = end - std::min(end, BUFFER_CHECK_INTERVAL);
            ensure_capacity(static_cast<size_t>(end - begin) * 4 + 16);
            for (py::ssize_t position = end; position-- > begin;)
                put(data[2 * position], data[2 * position + 1], PRECISION);
            end = begin;
        }
    }

    void encode_categorical_shared(SymbolArray symbols, CdfArray cdf) {
        if (cdf.ndim() != 1)
            throw std::runtime_error("Expected a single categorical CDF [K-1].");
        const auto *symbol_data = symbols.data();
        const auto *cdf_data = cdf.data();
        const uint32_t boundary_count = static_cast<uint32_t>(cdf.size());
        py::gil_scoped_release release;
        for (py::ssize_t end = symbols.size(); end > 0;) {
            const py::ssize_t begin = end - std::min(end, BUFFER_CHECK_INTERVAL);
            ensure_capacity(static_cast<size_t>(end - begin) * 4 + 16);
            for (py::ssize_t position = end; position-- > begin;) {
                const uint32_t symbol = static_cast<uint32_t>(symbol_data[position]);
                const uint32_t start = symbol == 0 ? 0 : cdf_data[symbol - 1];
                const uint32_t stop = symbol == boundary_count ? CDF_TOTAL : cdf_data[symbol];
                put(start, stop - start, PRECISION);
            }
            end = begin;
        }
    }

    py::bytes flush() {
        ensure_capacity(4);
        RansEncFlush(&state_, &output_);
        const char *data = reinterpret_cast<const char *>(output_);
        const size_t size = buffer_.get() + capacity_ - output_;
        py::bytes result(data, size);
        reset();
        return result;
    }

private:
    void reset() {
        output_ = buffer_.get() + capacity_;
        RansEncInit(&state_);
    }

    void ensure_capacity(size_t additional) {
        const size_t available = static_cast<size_t>(output_ - buffer_.get());
        if (available >= additional)
            return;
        const size_t encoded = buffer_.get() + capacity_ - output_;
        const size_t new_size = std::max(capacity_ * 2,
                                         encoded + additional + 1024);
        std::unique_ptr<uint8_t[]> replacement(new uint8_t[new_size]);
        std::memcpy(replacement.get() + new_size - encoded, output_, encoded);
        buffer_.swap(replacement);
        capacity_ = new_size;
        output_ = buffer_.get() + capacity_ - encoded;
    }

    void put(uint32_t start, uint32_t frequency, uint32_t precision) {
        if (frequency == 0)
            throw std::runtime_error("Cannot encode a zero-frequency symbol.");
        RansEncPut(&state_, &output_, start, frequency, precision);
    }

    void encode_bit(uint32_t bit) { put(bit, 1, 1); }

    void encode_exp_golomb(uint32_t value) {
        uint32_t quotient = value + 1;
        int bits = 0;
        while (quotient != 0) {
            encode_bit(quotient & 1u);
            quotient >>= 1;
            ++bits;
        }
        while (--bits > 0)
            encode_bit(0);
    }

    void encode_indexed_symbol(int32_t symbol, uint16_t table_index,
                               const CdfTables &tables) {
        const auto &cdf = tables.cdfs[table_index];
        int32_t value = symbol - tables.offsets[table_index];
        const int32_t escape = static_cast<int32_t>(cdf.size());
        if (value < 0 || value >= escape) {
            const uint32_t gamma = value < 0
                ? static_cast<uint32_t>(-value)
                : static_cast<uint32_t>(value - escape + 1);
            encode_bit(value < 0);
            encode_exp_golomb(gamma - 1);
            value = escape;
        }
        const uint32_t cdf_symbol = static_cast<uint32_t>(value);
        const uint32_t start = cdf_symbol == 0 ? 0 : cdf[cdf_symbol - 1];
        const uint32_t end = value == escape ? CDF_TOTAL : cdf[cdf_symbol];
        put(start, end - start, PRECISION);
    }

    size_t capacity_ = 1 << 20;
    std::unique_ptr<uint8_t[]> buffer_;
    uint8_t *output_ = nullptr;
    RansState state_ = 0;
};

class RansDecoder {
public:
    void set_stream(py::bytes encoded) {
        if (PyBytes_GET_SIZE(encoded.ptr()) < 4)
            throw std::runtime_error("rANS stream is too short.");
        stream_ = encoded;
        input_ = reinterpret_cast<const uint8_t *>(PyBytes_AS_STRING(stream_.ptr()));
        RansDecInit(&state_, &input_);
    }

    template <bool UseLookup>
    SymbolArray decode_indexed(IndexArray indexes, const CdfTables &tables) {
        SymbolArray symbols(indexes.size());
        decode_indexed_into<UseLookup>(indexes, tables, symbols);
        return symbols;
    }

    template <bool UseLookup>
    void decode_indexed_into(
        IndexArray indexes, const CdfTables &tables,
        py::array_t<int16_t, py::array::c_style> symbols) {
        const py::buffer_info index_info = indexes.request();
        if (symbols.size() != index_info.size)
            throw py::value_error("Output size must match index count");
        auto *output = symbols.mutable_data();
        const auto *index_data = static_cast<const uint16_t *>(index_info.ptr);
        py::gil_scoped_release release;
        for (py::ssize_t position = 0; position < index_info.size; ++position) {
            const uint16_t table_index = index_data[position];
            output[position] = static_cast<int16_t>(
                decode_indexed_symbol<UseLookup>(table_index, tables));
        }
    }

    SymbolArray decode_categorical(CdfArray cdfs, py::ssize_t count) {
        const py::buffer_info cdf_info = cdfs.request();
        if (cdf_info.ndim != 2 || (cdf_info.shape[0] != 1
            && cdf_info.shape[0] != count))
            throw std::runtime_error("Categorical CDF rows must be one or N.");
        SymbolArray symbols(count);
        auto *output = static_cast<int16_t *>(symbols.request().ptr);
        const bool shared_cdf = cdf_info.shape[0] == 1;
        const uint32_t boundary_count = static_cast<uint32_t>(cdf_info.shape[1]);
        const py::ssize_t row_stride = cdf_info.strides[0];
        py::gil_scoped_release release;
        for (py::ssize_t position = 0; position < count; ++position) {
            const py::ssize_t row = shared_cdf ? 0 : position;
            const auto *cdf = reinterpret_cast<const uint16_t *>(
                static_cast<const uint8_t *>(cdf_info.ptr) + row * row_stride);
            const uint32_t cumulative = RansDecGet(&state_, PRECISION);
            const auto *upper = std::upper_bound(cdf, cdf + boundary_count,
                                                  cumulative);
            const uint32_t symbol = static_cast<uint32_t>(upper - cdf);
            const uint32_t start = symbol == 0 ? 0 : cdf[symbol - 1];
            const uint32_t end = symbol == boundary_count
                                 ? CDF_TOTAL : cdf[symbol];
            RansDecAdvance(&state_, &input_, start, end - start, PRECISION);
            output[position] = static_cast<int16_t>(symbol);
        }
        return symbols;
    }

    SymbolArray decode_categorical_shared(CdfArray cdf, py::ssize_t count) {
        if (cdf.ndim() != 1)
            throw std::runtime_error("Expected a single categorical CDF [K-1].");
        SymbolArray symbols(count);
        auto *output = symbols.mutable_data();
        const auto *cdf_data = cdf.data();
        const uint32_t boundary_count = static_cast<uint32_t>(cdf.size());
        uint16_t inverse_cdf[CDF_TOTAL];
        py::gil_scoped_release release;
        uint32_t start = 0;
        for (uint32_t symbol = 0; symbol <= boundary_count; ++symbol) {
            const uint32_t end = symbol == boundary_count ? CDF_TOTAL : cdf_data[symbol];
            std::fill(inverse_cdf + start, inverse_cdf + end, static_cast<uint16_t>(symbol));
            start = end;
        }
        for (py::ssize_t position = 0; position < count; ++position) {
            const uint32_t symbol = inverse_cdf[RansDecGet(&state_, PRECISION)];
            const uint32_t begin = symbol == 0 ? 0 : cdf_data[symbol - 1];
            const uint32_t end = symbol == boundary_count ? CDF_TOTAL : cdf_data[symbol];
            RansDecAdvance(&state_, &input_, begin, end - begin, PRECISION);
            output[position] = static_cast<int16_t>(symbol);
        }
        return symbols;
    }

private:
    uint32_t decode_bit() {
        const uint32_t bit = RansDecGet(&state_, 1);
        RansDecAdvance(&state_, &input_, bit, 1, 1);
        return bit;
    }

    uint32_t decode_exp_golomb() {
        int prefix = 0;
        while (decode_bit() == 0)
            ++prefix;

        uint32_t quotient = 1u << prefix;
        for (int bit = prefix - 1; bit >= 0; --bit)
            quotient |= decode_bit() << bit;

        return quotient - 1;
    }

    template <bool UseLookup>
    int32_t decode_indexed_symbol(uint16_t table_index,
                                  const CdfTables &tables) {
        const auto &cdf = tables.cdfs[table_index];
        const uint32_t cumulative = RansDecGet(&state_, PRECISION);
        int32_t value;
        if constexpr (UseLookup) {
            const auto *inverse = tables.inverse_cdfs.data() + static_cast<size_t>(table_index) * CDF_LOOKUP_STRIDE;
            const uint32_t bucket = cumulative >> (PRECISION - CDF_LOOKUP_BITS);
            const auto upper = std::upper_bound(
                cdf.begin() + inverse[bucket], cdf.begin() + inverse[bucket + 1], cumulative);
            value = static_cast<int32_t>(upper - cdf.begin());
        } else {
            const auto upper = std::upper_bound(cdf.begin(), cdf.end(), cumulative);
            value = static_cast<int32_t>(upper - cdf.begin());
        }
        const uint32_t symbol = static_cast<uint32_t>(value);
        const uint32_t start = symbol == 0 ? 0 : cdf[symbol - 1];
        const uint32_t end = value == static_cast<int32_t>(cdf.size())
                             ? CDF_TOTAL : cdf[symbol];
        RansDecAdvance(&state_, &input_, start, end - start, PRECISION);

        const int32_t escape = static_cast<int32_t>(cdf.size());
        if (value != escape)
            return value + tables.offsets[table_index];

        const uint32_t gamma = decode_exp_golomb() + 1;
        const uint32_t sign = decode_bit();
        value = sign ? -static_cast<int32_t>(gamma)
                     : static_cast<int32_t>(gamma) + escape - 1;
        return value + tables.offsets[table_index];
    }

    py::bytes stream_;
    const uint8_t *input_ = nullptr;
    RansState state_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<CdfTables>(module, "CdfTables")
        .def(py::init<>())
        .def_readwrite("cdfs", &CdfTables::cdfs)
        .def_readwrite("offsets", &CdfTables::offsets);
    module.def(
        "quantize_pmfs", &quantize_pmfs,
        py::arg("pmfs"), py::arg("offsets"));
    module.def("build_inverse_cdfs", &build_inverse_cdfs, py::arg("tables"));
    py::class_<RansEncoder>(module, "RansEncoder")
        .def(py::init<>())
        .def("encode_indexed", &RansEncoder::encode_indexed)
        .def("encode_categorical", &RansEncoder::encode_categorical)
        .def("encode_categorical_shared", &RansEncoder::encode_categorical_shared)
        .def("flush", &RansEncoder::flush);
    py::class_<RansDecoder>(module, "RansDecoder")
        .def(py::init<>())
        .def("set_stream", &RansDecoder::set_stream)
        .def("decode_indexed", &RansDecoder::decode_indexed<false>)
        .def("decode_indexed_into", &RansDecoder::decode_indexed_into<false>,
             py::arg("indexes"), py::arg("tables"), py::arg("symbols").noconvert())
        .def("decode_indexed_lookup", &RansDecoder::decode_indexed<true>)
        .def("decode_indexed_lookup_into", &RansDecoder::decode_indexed_into<true>,
             py::arg("indexes"), py::arg("tables"), py::arg("symbols").noconvert())
        .def("decode_categorical", &RansDecoder::decode_categorical)
        .def("decode_categorical_shared", &RansDecoder::decode_categorical_shared);
}
