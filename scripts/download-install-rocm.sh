#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: scripts/download-install-rocm.sh


# wget https://repo.radeon.com/rocm/installer/rocm-runfile-installer/rocm-rel-<rocm-version>/<distro>/<distro-version>/<installer-file>

INSTALLER=https://repo.radeon.com/rocm/installer/rocm-runfile-installer/rocm-rel-6.4.1/ubuntu/22.04/rocm-installer_1.1.1.60401-30-83~22.04.run

set -euxo pipefail

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
    ./${installer_filename} --noexec
fi

cd ${installer_dir}

sed -i 's/ubuntu)/ubuntu|devuan)/' *.sh
sed -i 's/^'${nominal_distro_version}'$/'${actual_distro_version}/ VERSION

./install-init.sh "$@"
