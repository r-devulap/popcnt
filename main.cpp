#include "immintrin.h"
#include <iostream>
#include <cstdint>

// version 1: https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx512-vpopcnt.cpp
uint64_t avx512_vpopcnt(const uint8_t* data, const size_t size) {
    const size_t chunks = size / 64;

    uint8_t* ptr = const_cast<uint8_t*>(data);
    const uint8_t* end = ptr + size;

    // count using AVX512 registers
    __m512i accumulator = _mm512_setzero_si512();
    for (size_t i=0; i < chunks; i++, ptr += 64) {
        // Note: a short chain of dependencies, likely unrolling will be needed.
        const __m512i v = _mm512_loadu_si512((const __m512i*)ptr);
        const __m512i p = _mm512_popcnt_epi64(v);

        accumulator = _mm512_add_epi64(accumulator, p);
    }

    // horizontal sum of a register
    uint64_t tmp[8] __attribute__((aligned(64)));
    _mm512_store_si512((__m512i*)tmp, accumulator);

    uint64_t total = 0;
    for (size_t i=0; i < 8; i++) {
        total += tmp[i];
    }

    // popcount the tail
    while (ptr + 8 < end) {
        total += _mm_popcnt_u64(*reinterpret_cast<const uint64_t*>(ptr));
        ptr += 8;
    }

    // ignore this: buf size is always a multiple of 8
    //while (ptr < end) {
    //    total += lookup8bit[*ptr++];
    //}

    return total;
}


uint64_t avx512_vpopcnt_reduce(const uint8_t* data, const size_t size) {
    uint64_t popcnt = 0;
    for (size_t ii = 0; ii < size; ii = ii + 64) {
        __m512i buf = _mm512_loadu_si512(data + ii);
        popcnt += (uint64_t)_mm512_reduce_add_epi64(
            _mm512_popcnt_epi64(buf));
    }
    return popcnt;
}

#ifdef __MAIN__
int main() {
    size_t bufsize = 6400000;
    uint8_t* a = new uint8_t[bufsize];
    for (auto ii = 0; ii < bufsize; ++ii) {
        a[ii] = rand() % 128;
    }
    std::cout << "RD_DEBUG: " << avx512_vpopcnt_reduce(a, bufsize) << ", " << avx512_vpopcnt(a, bufsize) << std::endl;
    return 0;
}
#else
#include <benchmark/benchmark.h>
static void pocnt_accumulator(benchmark::State& state) {
    // Perform setup here
    size_t bufsize = state.range(0);
    uint8_t* a = new uint8_t[bufsize];
    for (auto _ : state) {
        uint64_t retval = avx512_vpopcnt(a, bufsize);
        benchmark::DoNotOptimize(retval);
    }
}

static void pocnt_reduce(benchmark::State& state) {
    // Perform setup here
    size_t bufsize = state.range(0);
    uint8_t* a = new uint8_t[bufsize];
    for (auto _ : state) {
        uint64_t retval = avx512_vpopcnt_reduce(a, bufsize);
        benchmark::DoNotOptimize(retval);
    }
}

// Register the function as a benchmark
BENCHMARK(pocnt_reduce)->Arg(64*1000);
BENCHMARK(pocnt_accumulator)->Arg(64*1000);
#endif
