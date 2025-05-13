cmake_minimum_required(VERSION 3.26)

project(SAF
    VERSION 0.01
    DESCRIPTION "Small Application Framework."
    LANGUAGES CXX C
)

set(CMAKE_COLOR_MAKEFILE ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug, Release" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install/saf/" CACHE PATH "Install path prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

option(SAF_BUILD_DOC "Build Documentation" OFF)
option(SAF_ENABLE_SCRIPTING "Enable Lua Scripting" OFF)
option(SAF_ENABLE_FILE_WATCH "Enable File Watching" OFF)
option(SAF_ENABLE_CUDA_INTEROP "Enable Easy Interoperability with CUDA" OFF)

set(VERSION_MAJOR 0 CACHE STRING "Project major version number.")
set(VERSION_MINOR 0 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 1 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

message(STATUS "================================================")
message(STATUS "             #####     #    #######             ")
message(STATUS "            #     #   # #   #                   ")
message(STATUS "            #        #   #  #                   ")
message(STATUS "             #####  #     # #####               ")
message(STATUS "                  # ####### #                   ")
message(STATUS "            #     # #     # #                   ")
message(STATUS "             #####  #     # #                   ")
message(STATUS "================================================")
message(STATUS "Copyright (c) 2023-present, Paul Himmler")
message(STATUS "================================================")
message(STATUS "Version v" ${VERSION_MAJOR} "." ${VERSION_MINOR} ${VERSION_PATCH} " Licensed Apache License 2.0")
message(STATUS "================================================")
message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "Setting Project Standards.")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)

if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

message(STATUS "================================================")

message(STATUS "Fetching Dependencies.")

# dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)
set(FETCHCONTENT_QUIET TRUE)

# Vulkan
find_package(Vulkan REQUIRED)

if(SAF_ENABLE_CUDA_INTEROP)
    # CUDA
    enable_language(CUDA)
endif()

# Eigen
include(eigen3)

# glfw
include(glfw)

# imgui and more
include(imgui)

if(SAF_ENABLE_SCRIPTING)
    # lua and sol2
    include(lua)
    include(sol2)
endif()

if(SAF_ENABLE_FILE_WATCH)
    # efsw
    include(efsw)
endif()

if(SAF_BUILD_DOC)
    find_package(Doxygen REQUIRED dot)

    if(DOXYGEN_FOUND)
        add_subdirectory(doc)
    else()
        message(WARNING "Doxygen not found, documentation will not be build. Please Install Doxygen to build the documentation!")
    endif()
endif()

message(STATUS "================================================")

set(SAF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/saf)

add_subdirectory(saf)

add_subdirectory(examples)

message(STATUS "================================================")
