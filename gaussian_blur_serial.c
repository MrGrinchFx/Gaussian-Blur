#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


unsigned char* read_pgm(const char* filename, int* width, int* height, int* maxval) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("cant open");
        exit(EXIT_FAILURE);
    }

    char header[3];
    if(!fscanf(file, "%s\n", header)){
        exit(EXIT_FAILURE);
    }
    if (strcmp(header, "P5") != 0) {
        fclose(file);
        fprintf(stderr, "oopsie\n");
        exit(EXIT_FAILURE);
    }

    if(!fscanf(file, "%d %d\n%d\n", width, height, maxval)){
        exit(EXIT_FAILURE);
    }
    unsigned char* data = (unsigned char*)malloc(*width * *height);
    if(!fread(data, 1, *width * *height, file)){
        exit(EXIT_FAILURE);
    }
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


void apply_gaussian_blur(unsigned char* input, unsigned char* output, int width, int height, float* kernel, int order) {
    int half_size = order / 2;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
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
