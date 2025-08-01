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


if [ "$#" -lt 1 ]; then
    target_dir="."
else
    target_dir="${1}"
    shift
fi

cloc_def_file="$(dirname $(dirname $(readlink -f "$0")))/cloc-def.txt"

if which cloc > /dev/null; then
    cloc --vcs git --read-lang-def "${cloc_def_file}" "${target_dir}" "$@"
else
    echo '[ERROR] `cloc` is required to execute this script. See https://github.com/AlDanial/cloc' 1>&2
fi
