cmake_minimum_required(VERSION 3.14)
project(quillan.cpp VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization and Architecture flags
if(MSVC)
    add_compile_options(/W3 /arch:AVX2 /MP)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        add_compile_options(/O2)
    endif()
else()
    add_compile_options(-O3 -Wall -Wextra -mavx2 -mfma)
endif()

# Include directories
include_directories(include)

# Core Library
add_library(quillan STATIC
    src/quillan.cpp
)

target_include_directories(quillan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CLI Application
add_executable(quillan-cli
    src/main.cpp
)
target_link_libraries(quillan-cli PRIVATE quillan)

# Test Suite
add_executable(test_engine
    tests/test_engine.cpp
)
target_link_libraries(test_engine PRIVATE quillan)

enable_testing()
add_test(NAME engine_parity_test COMMAND test_engine)

# Install targets
install(TARGETS quillan quillan-cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
