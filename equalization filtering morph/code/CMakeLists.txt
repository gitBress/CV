cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Lab03CV)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(filter
	filter.h	
	filter.cpp)

add_executable(${PROJECT_NAME} Lab03CV.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})