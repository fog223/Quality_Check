cmake_minimum_required(VERSION 3.26)
project(Quality_Check)

set(CMAKE_CXX_STANDARD 11)

find_package(PCL CONFIG REQUIRED)
find_package(libLAS CONFIG REQUIRED)

add_executable(Quality_Check main.cpp)

target_include_directories(Quality_Check SYSTEM PRIVATE ${libLAS_INCLUDE_DIRS})
target_link_libraries(Quality_Check PRIVATE ${PCL_LIBRARIES} $<$<PLATFORM_ID:Windows>:lib>las)