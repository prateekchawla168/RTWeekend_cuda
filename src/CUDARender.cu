#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <time.h>

// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

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

__global__ void ConvertInt(int *out, float *in, int nItems ) {

    // converts a float array to int
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < nItems) {
        out[i] = int(255.999 * in[i]);
    }


}


int CUDARender() {

    // set up image
    int nx = 1280;
    int ny = 720;
    int numChannels = 3;
    
    // set up threadblocks - we can launch 32x32 threads at a time
    int tx = 8;
    int ty = 8;

    std::clog << "Rendering a " << nx << "x" << ny << " image ";
    std::clog << "in " << tx << "x" << ty << " blocks.\n";

    int numPixels = nx * ny;
    size_t fbSize = numChannels * numPixels * sizeof(float);

    // allocate fb
    float* fb;
    CheckCudaErrors(cudaMallocManaged((void**)&fb, fbSize));


    clock_t start, stop;

    start = clock();
    // render the buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    Render <<< blocks, threads >>> (fb, nx, ny);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    stop = clock();

    double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::clog << "Render complete in " << timerSeconds << " sec.\n";

    unsigned char* imageData = new unsigned char[nx * ny * numChannels];
    /*
    std::ofstream outputFile("output/HelloWorld.ppm"); // open file
    outputFile << "P3\n" << nx << " " << ny << "\n255\n"; //write first line


    // output as image
    for (int j = 0; j < ny; j++) {
        std::clog << "\rLines Remaining : " << (ny - j - 1) << ' ' << std::flush; // quick progress indicator
        for (int i = 0; i < nx; i++) {
            
            size_t pixelIdx = (i * 3) + (j * 3 * nx);

            // read from memory
            auto r = fb[pixelIdx + 0]; 
            auto g = fb[pixelIdx + 1]; 
            auto b = fb[pixelIdx + 2]; 

            auto ir = int(255.999 * r);
            auto ig = int(255.999 * g);
            auto ib = int(255.999 * b);

            outputFile << ir << " " << ig << " " << ib << "\n";
            imageData[pixelIdx + 0] = ir;
            imageData[pixelIdx + 1] = ig;
            imageData[pixelIdx + 2] = ib;

        }
    }

    outputFile.close();
    */
    
    // stbi_write_png( filename_str, width, height, number_of_channels, pixel_data, stride )
    // stride = how many to skip until we get to the next row? 
    // In this case, stride = width * number_of_channels
    stbi_write_png("output/HelloWorld.png", nx, ny, numChannels, imageData, nx * numChannels);


    // remember to release memory!
    delete[] imageData;
    CheckCudaErrors(cudaFree(fb));
    std::clog << "\nDone!\n";

    return 0;
}
