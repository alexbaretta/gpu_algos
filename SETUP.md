# Setup Instructions

# Use homebrew to install the necessary dependencies

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
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

2. Add CUDA to your PATH, ideally in /etc/profile:
```bash
   cat > /etc/profile.d/cuda.sh <<EOF
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
```

## Eigen

1. Install Eigen:
   ```bash
   sudo apt install libeigen3-dev
   ```


# Copyright notice
Copyright (c) 2025 Alessandro Baretta
All rights reserved.
