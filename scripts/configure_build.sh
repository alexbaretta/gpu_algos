#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

set -euxo pipefail

# $(dirname ${0})/create_clangd_helper_files.sh
cmake -Wno-dev --preset=debug
cmake -Wno-dev --preset=release

sed 's/--options-file /@/g' builds/debug/compile_commands.json > compile_commands.json
