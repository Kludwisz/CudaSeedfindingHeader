#ifndef SEEDFINDING_CUH
#define SEEDFINDING_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <cstdint>
#include <chrono>
#include <cstdio>

// ------------------------------------------------------------------------------
// General-purpose utilities

__host__ static void gpuAssert(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d — %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)


static constexpr uint64_t LCG_JAVA_MULTIPLIER = 0x5deece66d;
static constexpr uint32_t LCG_JAVA_ADDEND = 11;
static constexpr uint64_t REGION_SEED_A = 341873128712ULL;
static constexpr uint64_t REGION_SEED_B = 132897987541ULL;

static constexpr uint64_t MASK(uint8_t bits) {
    return (UINT64_C(1) << bits) - 1;
}
static constexpr uint64_t MASK_48 = MASK(48);
static constexpr uint32_t MASK_32 = static_cast<uint32_t>(MASK(32));
static constexpr uint32_t MASK_16 = static_cast<uint32_t>(MASK(16));


struct Timer {
    double full_work_size;
    double current_percent;
    double percent_per_update;
    std::chrono::time_point<std::chrono::steady_clock> time_start;

    inline Timer(double full_work_size, double percent_per_update = 1.0) {
        this->full_work_size = full_work_size;
        this->percent_per_update = percent_per_update;
        this->current_percent = 0.0;
    }

    inline void start() {
        time_start = std::chrono::steady_clock::now();
        this->current_percent = 0.0;
    }

    inline void update_completion(double new_work_done) {
        std::chrono::time_point<std::chrono::steady_clock> time_current = std::chrono::steady_clock::now();
        double new_percent = std::round(full_work_size / new_work_done * 100.0 / percent_per_update) * percent_per_update;

        if (new_percent != current_percent) {
            current_percent = new_percent;
            double seconds_for_current = (time_current - time_start).count() * 1e-9;
            double seconds_for_full = seconds_for_current * (full_work_size / new_work_done);
            double seconds_left = seconds_for_full - seconds_for_current;
            std::fprintf(stderr, "----- %f %% done, ETA: %f seconds\n", current_percent, seconds_left);
        }
    }
};

// ------------------------------------------------------------------------------
// TODO Xoroshiro

// ------------------------------------------------------------------------------
// TODO Java Random

#endif // SEEDFINDING_CUH