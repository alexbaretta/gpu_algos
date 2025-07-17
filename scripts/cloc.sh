#!/bin/bash

# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

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
