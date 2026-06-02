#include "scipy/core.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <random>

static void BM_trapezoid(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> y(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);
    for (size_t i = 0; i < n; ++i) y[i] = dist(rng);
    for (auto _ : state) {
        double r = scipy::integrate::trapezoid(y.data(), n);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_trapezoid)->Range(1<<10, 1<<20);

static void BM_normpdf(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> x(n), r(n);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < n; ++i) x[i] = dist(rng);
    for (auto _ : state) {
        scipy::stats::norm_pdf(x.data(), r.data(), n);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_normpdf)->Range(1<<10, 1<<20);
BENCHMARK_MAIN();
