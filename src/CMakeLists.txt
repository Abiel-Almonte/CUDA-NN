cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_project)

# Specify the path to the LibTorch installation
set(CMAKE_PREFIX_PATH "./libtorch")

find_package(Torch REQUIRED)

add_executable(my_project torch_main.cpp)
target_link_libraries(my_project "${TORCH_LIBRARIES}")
set_property(TARGET my_project PROPERTY CXX_STANDARD 23)
