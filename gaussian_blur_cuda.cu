#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>


#define cuda_check(ret) _cuda_check((ret), __FILE__, __LINE__)
inline void _cuda_check(cudaError_t ret, const char *file, int line)
    {
        if (ret != cudaSuccess) {
            fprintf(stderr, "CudaErr: %s (%s:%d)\n", cudaGetErrorString(ret), file, line);
            exit(1);
        }
    }

unsigned char* read_pgm(const char* filename, int* width, int* height, int* maxval) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("cant open");
        exit(EXIT_FAILURE);
    }

    char header[3];
    fscanf(file, "%s\n", header);
    if (strcmp(header, "P5") != 0) {
        fclose(file);
        fprintf(stderr, "oopsie\n");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d\n%d\n", width, height, maxval);
    unsigned char* data = (unsigned char*)malloc(*width * *height);
    fread(data, 1, *width * *height, file);
    fclose(file);

    return data;
}

void write_pgm(const char* filename, const unsigned char* data, int width, int height, int maxval) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        exit(EXIT_FAILURE);
    }

    fprintf(file, "P5\n%d %d\n%d\n", width, height, maxval);
    fwrite(data, 1, width * height, file);
    fclose(file);
}


float* create_gaussian_kernel(int* order, float sigma) {
    int half_size = (int)(sigma * 3);
    *order = 2 * half_size + 1;

    float* kernel = (float*)malloc((*order) * (*order) * sizeof(float));
    float sum = 0.0;
    for (int x = -half_size; x <= half_size; ++x) {
        for (int y = -half_size; y <= half_size; ++y) {
            int idx = (x + half_size) * (*order) + (y + half_size);
            kernel[idx] = expf(-(x*x + y*y) / (2 * sigma * sigma));
            sum += kernel[idx];
        }
    }

    for (int i = 0; i < (*order) * (*order); ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}


__global__ void apply_gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height, float* kernel, int order) {
    int half_size = order / 2;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float sum = 0.0;

        for (int ki = -half_size; ki <= half_size; ++ki) {
            for (int kj = -half_size; kj <= half_size; ++kj) {
                int ni = i + ki;
                int nj = j + kj;

                if (ni < 0) ni = 0;
                if (nj < 0) nj = 0;
                if (ni >= height) ni = height - 1;
                if (nj >= width) nj = width - 1;

                int kernel_idx = (ki + half_size) * order + (kj + half_size);
                sum += input[ni * width + nj] * kernel[kernel_idx];
            }
        }

        output[i * width + j] = (unsigned char)sum;
    }
}

void apply_gaussian_blur(unsigned char* input, unsigned char* output, int width, int height, float* kernel, int order) {
    unsigned char *d_input, *d_output;
    float *d_kernel;
    size_t size = width * height * sizeof(unsigned char);
    size_t size_kernel = order * order * sizeof(float);

    cuda_check(cudaMalloc(&d_input, size));
    cuda_check(cudaMalloc(&d_output, size));
    cuda_check(cudaMalloc(&d_kernel, size_kernel));

    cuda_check(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_kernel, kernel, size_kernel, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    apply_gaussian_blur_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, d_kernel, order);

    cuda_check(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(d_input));
    cuda_check(cudaFree(d_output));
    cuda_check(cudaFree(d_kernel));
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        return -1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    float sigma = atof(argv[3]);

 

    int width, height, maxval;
    unsigned char* input_image = read_pgm(input_filename, &width, &height, &maxval);

    int order;
    float* kernel = create_gaussian_kernel(&order, sigma);
  

    unsigned char* output_image = (unsigned char*)malloc(width * height);

    apply_gaussian_blur(input_image, output_image, width, height, kernel, order);

    write_pgm(output_filename, output_image, width, height, maxval);

    free(input_image);
    free(output_image);
    free(kernel);

    return 0;
}
