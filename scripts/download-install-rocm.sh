#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta <alex@baretta.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# source path: scripts/download-install-rocm.sh


# wget https://repo.radeon.com/rocm/installer/rocm-runfile-installer/rocm-rel-<rocm-version>/<distro>/<distro-version>/<installer-file>

INSTALLER=https://repo.radeon.com/rocm/installer/rocm-runfile-installer/rocm-rel-6.4.1/ubuntu/22.04/rocm-installer_1.1.1.60401-30-83~22.04.run
ROC_HOME=/opt/rocm
PYTHONLIB=libpython3.13.so.1.0

set -euo pipefail

# installer_filename=$( (IFS=/; for i in ${INSTALLER}; do echo $i; done | tail -n1) )
installer_filename=$(awk -F/ '{print $NF}' <<<"${INSTALLER}")
installer_dir=rocm-installer
nominal_distro_version_extension=$(awk -F'~' '{print $NF}' <<<"${installer_filename}")
nominal_distro_version=${nominal_distro_version_extension/.run/}
actual_distro_version=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')


if ! [ -f ${installer_filename} ]; then
    wget ${INSTALLER}
fi
if ! [ -d ${installer_dir} ]; then
    chmod a+x ${installer_filename}
    ./${installer_filename} --noexec
fi

cd ${installer_dir}

sed -i 's/ubuntu)/ubuntu|devuan)/' *.sh
sed -i 's/^'${nominal_distro_version}'$/'${actual_distro_version}/ VERSION

sudo ./install-init.sh "$@"

# Workaround for the following error:
# rocgdb: error while loading shared libraries: amdpythonlib.so: cannot open shared object file: No such file or directory
# 1) Detect the conda environment
conda_env="$(readlink -f $(dirname $(which python))/..)"


libpython_src="${conda_env}/lib/${PYTHONLIB}"
libpython_dst="${ROCM_HOME}/lib/${PYTHONLIB}"
# 2) Create symlink to ROCm lib directory

echo "[INFO] creating libamdpython symlink: ${libpython_src} -> ${libpython_dst}"
rm -f "${libpython_dst}"
sudo ln -sf "${libpython_src}" "${libpython_dst}"
