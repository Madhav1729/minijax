cmake_minimum_required(VERSION 3.20)
project(minijax_cpp CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  find_package(Eigen3 QUIET NO_MODULE)
endif()
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Core PATHS /usr/include/eigen3)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

find_package(Z3 QUIET)
if(Z3_FOUND)
  if(NOT TARGET z3::z3)
    add_library(z3::z3 ALIAS z3::libz3)
  endif()
else()
  find_path(Z3_INCLUDE_DIR z3++.h PATHS /usr/include)
  find_library(Z3_LIB z3 PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
  add_library(z3::z3 INTERFACE IMPORTED)
  set_target_properties(z3::z3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Z3_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${Z3_LIB}")
endif()

find_program(LLVM_CONFIG NAMES llvm-config-18 llvm-config-19 llvm-config-17 llvm-config)
if(NOT LLVM_CONFIG)
  message(WARNING "llvm-config not found — building WITHOUT the JIT backend "
                  "(minijax_jit skipped). Install llvm-devel or create the env "
                  "from environment.yml to enable it.")
endif()

if(LLVM_CONFIG)
execute_process(COMMAND ${LLVM_CONFIG} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --libs core orcjit native irreader OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(LLVM_LIBS_LIST UNIX_COMMAND "${LLVM_LIBS}")
separate_arguments(LLVM_SYSTEM_LIBS_LIST UNIX_COMMAND "${LLVM_SYSTEM_LIBS}")
message(STATUS "LLVM include: ${LLVM_INCLUDE_DIR}")
message(STATUS "LLVM libs: ${LLVM_LIBS}")

add_library(llvm_deps INTERFACE)
target_include_directories(llvm_deps INTERFACE ${LLVM_INCLUDE_DIR})
target_link_directories(llvm_deps INTERFACE ${LLVM_LIB_DIR})
target_link_libraries(llvm_deps INTERFACE ${LLVM_LIBS_LIST} ${LLVM_SYSTEM_LIBS_LIST})
endif()

add_library(minijax_core
  src/tensor.cpp
  src/ir.cpp
  src/interp.cpp
  src/autodiff.cpp
  src/egraph.cpp
  src/opt.cpp
  src/verify.cpp
  src/vm.cpp
  src/memplan.cpp
  src/viz.cpp
)
target_include_directories(minijax_core PUBLIC include)
target_link_libraries(minijax_core PUBLIC Eigen3::Eigen z3::z3)
target_compile_options(minijax_core PRIVATE -Wall -Wextra -O2)

if(LLVM_CONFIG)
add_library(minijax_jit src/jit.cpp)
target_include_directories(minijax_jit PUBLIC include)
target_link_libraries(minijax_jit PUBLIC minijax_core llvm_deps)
target_compile_options(minijax_jit PRIVATE -Wall -Wextra -O2)
endif()

add_library(minijax_extra
  src/nn.cpp
  src/fuzz.cpp
  src/frontend/lexer.cpp
  src/frontend/parser.cpp
  src/frontend/lower.cpp
)
target_include_directories(minijax_extra PUBLIC include)
target_link_libraries(minijax_extra PUBLIC minijax_core)
target_compile_options(minijax_extra PRIVATE -Wall -Wextra -O2)

if(LLVM_CONFIG)
  target_link_libraries(minijax_extra PRIVATE minijax_jit)
  target_compile_definitions(minijax_extra PRIVATE MINIJAX_WITH_JIT=1
                            INTERFACE MINIJAX_WITH_JIT=1)
endif()

add_executable(minijax src/cli.cpp)
target_link_libraries(minijax PRIVATE minijax_core minijax_extra
  $<TARGET_NAME_IF_EXISTS:minijax_jit>)

enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES tests/*.cpp)
if(NOT LLVM_CONFIG)
  list(FILTER TEST_SOURCES EXCLUDE REGEX "jit_test\\.cpp$")
endif()
add_executable(minijax_tests ${TEST_SOURCES})
target_link_libraries(minijax_tests PRIVATE minijax_core minijax_extra
  $<TARGET_NAME_IF_EXISTS:minijax_jit> gtest gtest_main)
target_include_directories(minijax_tests PRIVATE include)
include(GoogleTest)
gtest_discover_tests(minijax_tests)

if(EXISTS ${CMAKE_SOURCE_DIR}/bench/bench.cpp)
  add_executable(minijax_bench bench/bench.cpp)
  target_link_libraries(minijax_bench PRIVATE minijax_core minijax_extra
    $<TARGET_NAME_IF_EXISTS:minijax_jit>)
endif()
