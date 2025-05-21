# Setup Instructions

## CUDA Toolkit

1. Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Add CUDA to your PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## CMake

1. Download and install CMake from [cmake.org](https://cmake.org/download/)
2. Add CMake to your PATH:
   ```bash
   export PATH=/usr/local/cmake/bin:$PATH
   ```

## GCC

1. Install GCC 14:
   ```bash
   sudo apt-get install gcc-14 g++-14
   ```

## Eigen

1. Install Eigen:
   ```bash
   sudo apt-get install libeigen3-dev
   ```

# Host setup instruction


## Install CUDA

``` bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-debian12-12-9-local_12.9.0-575.51.03-1_amd64.deb
```

``` bash
# As root
dpkg -i cuda-repo-debian12-12-9-local_12.9.0-575.51.03-1_amd64.deb
cp /var/cuda-repo-debian12-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-9
```

# Copyright notice
Copyright (c) 2025 Alessandro Baretta
All rights reserved.
