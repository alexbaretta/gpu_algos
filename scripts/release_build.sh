#!/bin/bash

cmake --build --preset release -j $(nproc) "$@"
