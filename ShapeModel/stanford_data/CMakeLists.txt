cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(ssd_dataset)
find_package(OpenCV REQUIRED)

# Define source files
set(SOURCE_FILES
    data_builder.cpp
    data_builder.h
    helpers.cpp 
    helpers.h 
    sdd_parser.cpp 
    sdd_parser.h 
    structs.h 
)

# Add executable
add_executable(data_builder ${SOURCE_FILES})
target_link_libraries(data_builder ${OpenCV_LIBS})