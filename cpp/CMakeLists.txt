cmake_minimum_required(VERSION 3.16)
project(minijax_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(minijax_cpp
    src/main.cpp
)

target_include_directories(minijax_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()

add_executable(graph_smoke_test tests/smoke_test.cpp)
target_include_directories(graph_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME graph_smoke_test COMMAND graph_smoke_test)

add_executable(phase1_test tests/phase1_test.cpp)
target_include_directories(phase1_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME phase1_test COMMAND phase1_test)

add_executable(phase2_test tests/phase2_test.cpp)
target_include_directories(phase2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME phase2_test COMMAND phase2_test)

add_executable(gradcheck_test tests/gradcheck.cpp)
target_include_directories(gradcheck_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME gradcheck_test COMMAND gradcheck_test)

add_executable(opt_test tests/opt_test.cpp)
target_include_directories(opt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME opt_test COMMAND opt_test)

add_executable(flatten_test tests/flatten_test.cpp)
target_include_directories(flatten_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME flatten_test COMMAND flatten_test)

add_executable(extract_test tests/extract_test.cpp)
target_include_directories(extract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME extract_test COMMAND extract_test)

add_executable(complex_flatten_test tests/complex_flatten_test.cpp)
target_include_directories(complex_flatten_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME complex_flatten_test COMMAND complex_flatten_test)

add_executable(broadcast_flatten_test tests/broadcast_flatten_test.cpp)
target_include_directories(broadcast_flatten_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME broadcast_flatten_test COMMAND broadcast_flatten_test)

add_executable(flops_extract_test tests/flops_extract_test.cpp)
target_include_directories(flops_extract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME flops_extract_test COMMAND flops_extract_test)

add_executable(microbench tests/microbench.cpp)
target_include_directories(microbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# microbench is not added as a ctest, run manually

add_executable(microbench_large tests/microbench_large.cpp)
target_include_directories(microbench_large PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# microbench_large is not added as a ctest, run manually
