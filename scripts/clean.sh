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


set -euxo pipefail

find ./include/ \( -name '*.cpp' -o -name '*.cu' -o -name '*.hip' \) -delete

rm -rf builds
rm -f compile_commands.json
rm -rf .cache
