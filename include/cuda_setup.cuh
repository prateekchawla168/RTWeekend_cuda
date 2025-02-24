#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define CheckCudaErrors(val) CheckCuda( (val), #val, __FILE__, __LINE__ )

void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void Render(float* fb, int maxX, int maxY) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if ((i >= maxX) || (j >= maxY)) return;
    int pixelIdx = (j * maxX * 3) + (i * 3);
    fb[pixelIdx + 0] = float(i) / maxX;
    fb[pixelIdx + 1] = float(j) / maxY;
    fb[pixelIdx + 2] = 0.2;
}

__global__ void ConvertFloatToInt(unsigned char* intImage, float *floatImage, int imageSize){

    // int i = threadIdx.x + blockDim.x * blockIdx.x;
    // int j = threadIdx.y + blockDim.y * blockIdx.y;

    // if ((i >= dimX) || (j >= dimY)) return;
    // int pixelIdx = (j * dimY * 3) + (i * 3);

    // intImage[pixelIdx + 0] = int(255.999 * floatImage[pixelIdx + 0]);
    // intImage[pixelIdx + 1] = int(255.999 * floatImage[pixelIdx + 1]);
    // intImage[pixelIdx + 2] = int(255.999 * floatImage[pixelIdx + 2]);

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= imageSize) return;
    intImage[i] = __float2uint_rd(255.999 * floatImage[i]);

}