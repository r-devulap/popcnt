#include <immintrin.h>
#include <benchmark/benchmark.h>
#include <cmath>

inline static
__attribute__((target("avx512bw", "avx512f")))
double cosine_distance(const float* __restrict__ ax, const float* __restrict__ bx, int dim)
{
	float		distance = 0.0;
	float		norma = 0.0;
	float		normb = 0.0;
	double		similarity;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		distance += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	similarity = (double) distance / sqrt((double) norma * (double) normb);
	return similarity;
}

inline static
__attribute__((target("avx512bw", "avx512f")))
double cosine_avx512(const float* __restrict__ ax, const float* __restrict__ bx, int dim)
{
    __m512 ab = _mm512_set1_ps(0);
    __m512 aa = _mm512_set1_ps(0);
    __m512 bb = _mm512_set1_ps(0);

    __m512 a, b;
    __mmask16 loadmask;
    for (int i = 0; i < dim; i += 16) {
        if (dim - i < 16) {
            loadmask = ((1u << (dim - i)) - 1u);
            a = _mm512_maskz_loadu_ps(loadmask, ax + i);
            b = _mm512_maskz_loadu_ps(loadmask, bx + i);
        }
        else {
            a = _mm512_loadu_ps(ax + i);
            b = _mm512_loadu_ps(bx + i);
        }
        ab = _mm512_fmadd_ps(a, b, ab);
        aa = _mm512_fmadd_ps(a, a, aa);
        bb = _mm512_fmadd_ps(b, b, bb);
    }

    float abf = _mm512_reduce_add_ps(ab);
    float aaf = _mm512_reduce_add_ps(aa);
    float bbf = _mm512_reduce_add_ps(bb);
    return (double) abf / sqrt((double) aaf * (double) bbf);
}

static void cosine_autovec(benchmark::State &state)
{
    size_t bufsize = state.range(0);
    float * a = new float[bufsize];
    float * b = new float[bufsize];
    srand(42);
    for (size_t ii = 0; ii < bufsize; ++ii) {
        a[ii] = rand() / RAND_MAX;
        b[ii] = rand() / RAND_MAX;
    }
    for (auto _ : state) {
        auto retval = cosine_distance(a, b, bufsize);
        benchmark::DoNotOptimize(retval);
    }
}

static void cosine_simsimd(benchmark::State &state)
{
    size_t bufsize = state.range(0);
    float * a = new float[bufsize];
    float * b = new float[bufsize];
    srand(42);
    for (size_t ii = 0; ii < bufsize; ++ii) {
        a[ii] = rand() / RAND_MAX;
        b[ii] = rand() / RAND_MAX;
    }
    for (auto _ : state) {
        float retval = cosine_avx512(a, b, bufsize);
        benchmark::DoNotOptimize(retval);
    }
}

// Register the function as a benchmark
#define BENCH(func) \
    BENCHMARK(func)->Arg(64)->Arg(640)->Arg(6400)->Arg(64000);

BENCH(cosine_autovec)
BENCH(cosine_simsimd)
