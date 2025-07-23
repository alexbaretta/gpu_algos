#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: scripts/release_build.sh

set -euxo pipefail

project_root="$(dirname ${0})/.."
cd "$(readlink -f ${project_root})"

cmake --build --preset release -j $(nproc) "$@"
