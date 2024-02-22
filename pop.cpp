#include <benchmark/benchmark.h>
#include "immintrin.h"
#include <iostream>
#include <cstdint>

float avx512pd_sum(float *data, size_t N) {
  __m512d counter = _mm512_setzero_pd();
  for (size_t i = 0; i < N; i += 16) {
    __m512 v = _mm512_loadu_ps((__m512 *)&data[i]);
    __m512d part1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v, 0));
    __m512d part2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v, 1));
    counter = _mm512_add_pd(counter, part1);
    counter = _mm512_add_pd(counter, part2);
  }
  double sum = _mm512_reduce_add_pd(counter);
  for (size_t i = N / 16 * 16; i < N; i++) {
    sum += data[i];
  }
  return sum;
}

float avx512ps_sum(float *data, size_t N) {
  __m512 counter = _mm512_setzero_ps();
  for (size_t i = 0; i < N; i += 16) {
    __m512 v = _mm512_loadu_ps((__m512 *)&data[i]);
    counter = _mm512_add_ps(counter, v);
  }
  float sum = _mm512_reduce_add_ps(counter);
  for (size_t i = N / 16 * 16; i < N; i++) {
    sum += data[i];
  }
  return sum;
}

static void avx512pd(benchmark::State& state) {
    // Perform setup here
    size_t bufsize = state.range(0);
    float * a = new float[bufsize];
    srand(42);
    for (size_t ii = 0; ii < bufsize; ++ii) {
        a[ii] = rand() / RAND_MAX;
    }
    for (auto _ : state) {
        float retval = avx512pd_sum(a, bufsize);
        benchmark::DoNotOptimize(retval);
    }
}

static void avx512ps(benchmark::State& state) {
    // Perform setup here
    size_t bufsize = state.range(0);
    float * a = new float[bufsize];
    srand(42);
    for (size_t ii = 0; ii < bufsize; ++ii) {
        a[ii] = rand() / RAND_MAX;
    }
    for (auto _ : state) {
        float retval = avx512ps_sum(a, bufsize);
        benchmark::DoNotOptimize(retval);
    }
}

// Register the function as a benchmark
#define BENCH(func) \
    BENCHMARK(func)->Arg(64*10000000);

BENCH(avx512ps)
BENCH(avx512pd)
