#!/bin/bash

echo '[INFO] Create clangd helper source files for C++'
for i in $(find include/ -name '*.hpp'); do
    src_path=${i/.hpp/.cpp}
    src_file=$(basename ${src_path})
    # if find ./src -name ${src_file} | grep . > /dev/null; then
    #     echo " - ${src_path}"
    # else
    echo " * ${src_path}"
    cat > ${src_path} <<EOF
/* The purpose of this source file is to provide clangd with a compile command for the corresponding header file. */

#include "$(basename ${i})"
EOF
    # fi
done

echo '[INFO] Create clangd helper source files for CUDA'
for i in $(find include/ -name '*.cuh'); do
    src_path=${i/.cuh/.cu}
    src_file=$(basename ${src_path})
    # if find ./src -name ${src_file} | grep . > /dev/null; then
    #     echo " - ${src_path}"
    # else
    echo " * ${src_path}"
    cat > ${src_path} <<EOF
/* The purpose of this source file is to provide clangd with a compile command for the corresponding header file. */

#include "$(basename ${i})"
EOF
    # fi
done

echo '[INFO] Create clangd helper source files for HIP'
for i in $(find include/ -name '*.hiph'); do
    src_path=${i/.hiph/.hip}
    src_file=$(basename ${src_path})
    # if find ./src -name ${src_file} | grep . > /dev/null; then
    #     echo " - ${src_path}"
    # else
    echo " * ${src_path}"
    cat > ${src_path} <<EOF
/* The purpose of this source file is to provide clangd with a compile command for the corresponding header file. */

#include "$(basename ${i})"
EOF
    # fi
done
