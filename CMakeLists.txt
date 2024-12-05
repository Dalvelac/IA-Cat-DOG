cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ImageClassification C)

# Include directories for header files
include_directories(include)

# Add executable
add_executable(ImageClassification
        main.c
        src/image_data.c
        src/image_generator.c
        src/model.c
        src/train.c
)
