# CUDA Matrix Multiplication

This project demonstrates matrix multiplication using CUDA, targeting compute capability 8.9 and CUDA 12.9.

## Project Structure
```
.
├── CMakeLists.txt
├── include/         # Header files
├── src/            # Source files
│   └── main.cu     # Main CUDA program
├── build/          # Build directory
└── tmp/            # Temporary files
```

## Requirements
- CUDA Toolkit 12.9
- CMake 3.18 or higher
- C++17 compatible compiler
- NVIDIA GPU with compute capability 8.9

## Building the Project

1. Create and enter the build directory:
```bash
mkdir -p build
cd build
```

2. Configure and build:
```bash
cmake ..
make
```

## Running the Program

After building, run the program from the build directory:
```bash
./matrix_multiply
```

The program will:
1. Initialize two matrices (1000x10000 and 10000x1000)
2. Perform matrix multiplication on the GPU
3. Print the execution time
