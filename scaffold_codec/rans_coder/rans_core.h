// Minimal byte-aligned rANS core adapted from ryg_rans (CC0).
// https://github.com/rygorous/ryg_rans

#ifndef RANS_BYTE_H_
#define RANS_BYTE_H_

#include <cstdint>

using RansState = uint32_t;

constexpr uint32_t RANS_BYTE_L = 1u << 23;

inline void RansEncInit(RansState *state) {
    *state = RANS_BYTE_L;
}

inline void RansEncPut(RansState *state, uint8_t **output,
                       uint32_t start, uint32_t frequency,
                       uint32_t precision) {
    RansState value = *state;
    const uint32_t maximum = ((RANS_BYTE_L >> precision) << 8) * frequency;
    while (value >= maximum) {
        *--(*output) = static_cast<uint8_t>(value & 0xff);
        value >>= 8;
    }
    *state = ((value / frequency) << precision)
             + value % frequency + start;
}

inline void RansEncFlush(RansState *state, uint8_t **output) {
    const uint32_t value = *state;
    *output -= 4;
    (*output)[0] = static_cast<uint8_t>(value);
    (*output)[1] = static_cast<uint8_t>(value >> 8);
    (*output)[2] = static_cast<uint8_t>(value >> 16);
    (*output)[3] = static_cast<uint8_t>(value >> 24);
}

inline void RansDecInit(RansState *state, const uint8_t **input) {
    const uint8_t *ptr = *input;
    *state = static_cast<uint32_t>(ptr[0])
             | static_cast<uint32_t>(ptr[1]) << 8
             | static_cast<uint32_t>(ptr[2]) << 16
             | static_cast<uint32_t>(ptr[3]) << 24;
    *input += 4;
}

inline uint32_t RansDecGet(RansState *state, uint32_t precision) {
    return *state & ((1u << precision) - 1);
}

inline void RansDecAdvance(RansState *state, const uint8_t **input,
                           uint32_t start, uint32_t frequency,
                           uint32_t precision) {
    const uint32_t mask = (1u << precision) - 1;
    uint32_t value = frequency * (*state >> precision)
                     + (*state & mask) - start;
    while (value < RANS_BYTE_L)
        value = (value << 8) | *(*input)++;
    *state = value;
}

#endif
