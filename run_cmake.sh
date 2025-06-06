#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

set -euxo pipefail

cmake --preset=debug
cmake --preset=release

sed 's/--options-file /@/g' builds/debug/compile_commands.json > compile_commands.json
