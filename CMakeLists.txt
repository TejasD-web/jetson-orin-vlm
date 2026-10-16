cmake_minimum_required(VERSION 3.20)
project(edge-cloud-tools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- Options ----------
option(ENABLE_ASAN    "Enable AddressSanitizer"  OFF)
option(ENABLE_COVERAGE "Enable gcov coverage"    OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0)
    add_link_options(--coverage)
endif()

# ---------- Subdirectories ----------
add_subdirectory(tools)
