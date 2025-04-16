cmake_minimum_required(VERSION 3.16)
project(llm_inference_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# llama.cpp as subdirectory
add_subdirectory(llama)

# Your target
add_executable(inference_test src/inference.cpp)
add_executable(batch_inference_test src/batch_inference.cpp)

# Link with llama static library
target_include_directories(inference_test PRIVATE llama)
target_link_libraries(inference_test PRIVATE llama)

target_include_directories(batch_inference_test PRIVATE llama)
target_link_libraries(batch_inference_test PRIVATE llama)