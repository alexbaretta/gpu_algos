#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

set -euxo pipefail

find ./include/ \( -name '*.cpp' -o -name '*.cu' -o -name '*.hip' \) -delete

rm -rf builds
rm -f compile_commands.json
rm -rf .cache
