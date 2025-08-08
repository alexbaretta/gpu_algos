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

set -euo pipefail

project_root="$(dirname ${0})/.."
cd "$(readlink -f ${project_root})"

SED_PROGRAM=scripts/cuda_to_hip.sed

CUDA_HEADER_EXT=cuh
HIP_HEADER_EXT=hiph
CUDA_SRC_EXT=cu
HIP_SRC_EXT=hip


# convert include/**/*.cu* -> include/**/*.hip*
# convert src/**/*.cu* -> src/**/*.hip*

for cuda_file in $(find include src -name "*.${CUDA_HEADER_EXT}" -o -name "*.${CUDA_SRC_EXT}"); do
    # cuda_file is the path of the cuda source or header file:
    # We need to change any occurrence of `cuda` to `hip`, and
    # we need to change the suffix `.cu` to `.hip`. This implicitly converts `.cuh` -> `.hiph`

    # We need to perform pattern substitution twice, so we use `hip_header` as a temporary value...
    hip_file="$(echo ${cuda_file} | sed s/cuda/hip/g)"

    # ...then we store the final result in it.
    hip_file="${hip_file/.${CUDA_SRC_EXT}/.${HIP_SRC_EXT}}"
    hip_dir="$(dirname ${hip_file})"

    mkdir -p "${hip_dir}"

    echo sed -f "${SED_PROGRAM}" "${cuda_file}" \> "${hip_file}"
    sed -f "${SED_PROGRAM}" "${cuda_file}" > "${hip_file}"
done
