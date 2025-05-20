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
