#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: scripts/debug_build.sh

set -euxo pipefail

project_root="$(dirname ${0})/.."
cd "$(readlink -f ${project_root})"

cmake --build --preset debug -j $(nproc) "$@"
