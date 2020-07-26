cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(FinalProject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(FindPlate
	FindPlate.h	
	FindPlate.cpp)

add_library(FindChar
	FindChar.h	
	FindChar.cpp)

add_library(Classifier
	Classifier.h	
	Classifier.cpp)
	
add_executable(${PROJECT_NAME} FinalProject.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} FindPlate)
target_link_libraries(${PROJECT_NAME} FindChar)
target_link_libraries(${PROJECT_NAME} Classifier)