# Setup Instructions

## Copyright notice
Copyright (c) 2025 Alessandro Baretta

All rights reserved.

<!-- source path: SETUP.md -->

## Initialize and update submodules
```bash
git submodule update --init --recursive
```

## Development environment
This project uses a combination of C++, CUDA, and HIP programming languages. In order to get useful
syntax highlighting, code navigation, hover tooltip, and compiler errors and warnings, you need to
use clangd in conjunction with your IDE. For example, clangd is available as an extension for vscode.
Unfortunately, support for CUDA and HIP in the LLVM project's stock clangd is very limited, so you
should use clangd from the following fork of LLVM, which correctly detects CUDA and HIP programming
for header files based on their extensions.

https://github.com/alexbaretta/llvm-project

## Use homebrew to install the necessary dependencies

1. Install homebrew:
```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. Follow homebrews "Next steps" instructions
```
   ==> Next steps:
   - Run these commands in your terminal to add Homebrew to your PATH:
      echo >> /home/alex/.bashrc
      echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/alex/.bashrc
      eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
   - Install Homebrew's dependencies if you have sudo access:
      sudo apt-get install build-essential
   For more information, see:
      https://docs.brew.sh/Homebrew-on-Linux
   - We recommend that you install GCC:
      brew install gcc
   - Run brew help to get started
   - Further documentation:
      https://docs.brew.sh
```
3. Install packages
```
   brew install gcc@14 cmake
```

## CUDA Toolkit

1. Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
As normal user:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
```

As root user:
```bash
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt -y install cuda-toolkit-12-9
```

We also need install an older version of cuda to support clangd.
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
chmod a+x cuda_12.8.0_570.86.10_linux.run
```

As root user:
```bash
./cuda_12.8.0_570.86.10_linux.run
```


2. Add CUDA to your PATH, ideally in /etc/profile:
```bash
   cat > /etc/profile.d/cuda.sh <<EOF
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
```

## Eigen

Install Eigen:
```bash
sudo apt install libeigen3-dev
```

# ROCm

The instructions work for Debian Stable (at the time of this writing: Debian 12 Bookworm). Confirm the
version of the package to be installed on the AMD website

(https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html)

1. As normal user:
```bash
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.60401-1_all.deb
```

2. As root user:
```bash
apt update
apt install clang-19 clang-tidy-19 clangd-19 clang-tools-19 libomp-19-dev libstdc++-13-dev
apt install python3-setuptools python3-wheel
apt install ./amdgpu-install_6.4.60401-1_all.deb
```

3. Optional: Add support for Devuan (a Debian clone without systemd)
```bash
cp /usr/bin/amdgpu-install /usr/bin/amdgpu-install.bak
sed  -i 's/|debian/|debian|devuan/' /usr/bin/amdgpu-install
```

4. Continue installing as root:

If you have an AMD GPU:
```bash
amdgpu-install -y --usecase=workstation,rocm,rocmdev,mllib,mlsdk,dkms,graphics
modprobe amdgpu
```

If you do NOT have an AMD GPU but want to compile ROCm code anyway:
```bash
amdgpu-install -y --usecase=workstation,rocm,rocmdev,mllib,mlsdk
modprobe amdgpu
```


5. As root, add users to appropriate groups to have permission to use the GPU
```bash
usermod -a -G render,video [username]
```

6. As normal user, check installation and configuration
```bash
dkms status
```
The above command should report something like this:
```
amdgpu/x.x.x-xxxxxxx.xx.xx, x.x.x-xx-generic, x86_64: installed
```

```bash
rocminfo
```
The above command should list the GPU:
```
[...]
*******
Agent 2
*******
  Name:                    gfx1201
  Uuid:                    GPU-6952ae06c4eeebab
  Marketing Name:          AMD Radeon RX 9070
  Vendor Name:             AMD
  [...]
[...]
```

```bash
clinfo
```
The above command should list the GPU:
```
Number of platforms:                             1
  Platform Profile:                              FULL_PROFILE
  Platform Version:                              OpenCL 2.1 AMD-APP (3649.0)
  Platform Name:                                 AMD Accelerated Parallel Processing
  Platform Vendor:                               Advanced Micro Devices, Inc.
  Platform Extensions:                           cl_khr_icd cl_amd_event_callback


  Platform Name:                                 AMD Accelerated Parallel Processing
Number of devices:                               1
  Device Type:                                   CL_DEVICE_TYPE_GPU
  Vendor ID:                                     1002h
  Board name:                                    AMD Radeon RX 9070
  [...]
[...]
```

7. (Optional) Install the ROCm CMake build tools

The ROCm CMake build tools are useful to build ROCm from source, not to build a project that uses ROCm.

```bash
git clone https://github.com/ROCm/rocm-cmake.git
(
   set -euxo pipefail
   cd rocm-cmake
   mkdir -p build
   cd build
   cmake ..
   cmake --build .
   sudo cmake --build . --target install
)
```

8. Set up user-level clangd configuration (pay attention not to overwrite your own config file with the following command.)
```bash
mkdir -p ~/.config/clangd
cp .clangd ~/.config/clangd/config.yaml
```

## librt linking errors

librt is now included in glibc, so we no longer need CUDA to link it. Until a new version of cmake is
released that can handle this new oddity, it is best to manualy edit FindCUDAToolkit.cmake as follows:

```cmake
      #find_library(CUDAToolkit_rt_LIBRARY rt)
      #mark_as_advanced(CUDAToolkit_rt_LIBRARY)
      #if(NOT CUDAToolkit_rt_LIBRARY)
      #  message(WARNING "Could not find librt library, needed by CUDA::cudart_static")
      #else()
      #  target_link_libraries(CUDA::cudart_static_deps INTERFACE ${CUDAToolkit_rt_LIBRARY})
      #endif()
```
