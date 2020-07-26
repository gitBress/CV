cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(Lab05CV)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(PanoramicImage
	PanoramicImage.h	
	PanoramicImage.cpp)
	
add_executable(${PROJECT_NAME} Lab05CV.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PanoramicImage)
