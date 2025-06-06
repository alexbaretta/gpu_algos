# Copyright (c) 2025 Alessandro Baretta
# All rights reserved.

# source path: src/benchmarks/CMakeLists.txt


# Auto-discover and include all technology subdirectories (cuda, hip, opencl, etc.)
file(GLOB tech_dirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(tech_dir ${tech_dirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${tech_dir})
        # Look for tool subdirectories within each technology directory
        file(GLOB tool_dirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${tech_dir} ${CMAKE_CURRENT_SOURCE_DIR}/${tech_dir}/*)
        foreach(tool_dir ${tool_dirs})
            if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${tech_dir}/${tool_dir})
                if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${tech_dir}/${tool_dir}/CMakeLists.txt)
                    message(STATUS "Including tool: ${tech_dir}/${tool_dir}")
                    add_subdirectory(${tech_dir}/${tool_dir})
                endif()
            endif()
        endforeach()
    endif()
endforeach()
