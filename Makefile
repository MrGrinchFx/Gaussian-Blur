# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

# Source files
SRCS_SERIAL = gaussian_blur_serial.c
SRCS_CUDA = gaussian_blur_cuda.cu

# Object files
OBJS_SERIAL = $(SRCS_SERIAL:.c=.o)
OBJS_CUDA = $(SRCS_CUDA:.cu=.o)

# Executable names
TARGET_SERIAL = gaussian_blur_serial
TARGET_CUDA = gaussian_blur_cuda

# Default target
all: $(TARGET_SERIAL) $(TARGET_CUDA)

# Rule to build the serial version
$(TARGET_SERIAL): $(OBJS_SERIAL)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build the CUDA version
$(TARGET_CUDA): $(OBJS_CUDA)
	$(NVCC) -o $@ $^

# Rule to compile serial C files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile CUDA C files to object files
%.o: %.cu
	$(NVCC) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS_SERIAL) $(OBJS_CUDA) $(TARGET_SERIAL) $(TARGET_CUDA)

# Phony targets
.PHONY: all clean

