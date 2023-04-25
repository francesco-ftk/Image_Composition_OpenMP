cmake_minimum_required(VERSION 3.20)
project(Image_Composition_OpenMP)

set(CMAKE_CXX_STANDARD 17)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR C:/Development/OpenCV/build)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(Image_Composition_OpenMP main.cpp)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

# linking
target_link_libraries(Image_Composition_OpenMP ${OpenCV_LIBS} OpenMP::OpenMP_CXX)