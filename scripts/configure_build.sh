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

BUILD_PYTHON_PACKAGE=ON
while [ $# -gt 0 ]; do
    case "${1}" in
        (--no-python) BUILD_PYTHON_PACKAGE=OFF; shift;;
        (--python) BUILD_PYTHON_PACKAGE=ON; shift;;
        (*) echo "[ERROR] unrecognized option: ${1}"; exit 1;;
    esac
done


# $(dirname ${0})/create_clangd_helper_files.sh
cmake -Wno-dev --preset=debug -DBUILD_PYTHON_PACKAGE=ON "$@"
cmake -Wno-dev --preset=release -DBUILD_PYTHON_PACKAGE=ON "$@"

sed 's/--options-file /@/g' builds/debug/compile_commands.json > compile_commands.json
