#include "seedfinding.cuh"


__global__ void kern(uint64_t offset) {
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint64_t rand = tid + offset;
    for (int i = 0; i < 110; i++) {
        if (nextFloat(&rand) < 0.2f) {
            return;
        }
    }

    printf("%u\n", tid);
}

int main() {
    int batches = 32;

    Timer t(batches, 10.0);
    t.start();
    for (int b = 0; b < batches; b++) {
        uint32_t n_blocks = (1ULL << 32) / 256;
        uint32_t threads_pb = 256;
        kern<<< n_blocks, threads_pb >>>(b * (1ULL << 32));
        CUDA_CHECK(cudaDeviceSynchronize());
        t.update_completion(b);
    }
}