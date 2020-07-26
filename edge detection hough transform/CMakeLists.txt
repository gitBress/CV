cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(Lab04CV)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} Lab04CV.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})