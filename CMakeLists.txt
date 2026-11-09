cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
endif()
project(cup_phase LANGUAGES CXX CUDA)

foreach(example d3q19_phase_tr d3q19_phase_tr_force)
    add_library(${example} SHARED ${example}/cuda_core.cu)
    target_include_directories(${example} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${example} PRIVATE $<$<PLATFORM_ID:Windows>:WIN32> MP_LIBRARY)
    set_target_properties(${example} PROPERTIES
        CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
        OUTPUT_NAME cuda_core
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${example}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${example}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${example}")
endforeach()

add_custom_command(TARGET d3q19_phase_tr_force POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/d3q19_phase_tr_force/ini"
        "$<TARGET_FILE_DIR:d3q19_phase_tr_force>/ini")
