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
set(FETCHCONTENT_QUIET TRUE)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)

cmake_language(GET_MESSAGE_LOG_LEVEL CMAKE_MESSAGE_LOG_LEVEL_BACKUP)
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
set(FETCHCONTENT_QUIET ON)

# Vulkan
find_package(Vulkan REQUIRED)

# imgui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.90.4-docking
    GIT_SHALLOW ON
    GIT_PROGRESS ON
)

# Eigen
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    GIT_VERBOSE ON
)

# glfw
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG 3.3.8
    GIT_SHALLOW ON
    GIT_PROGRESS ON
)

# pybind11
# FetchContent_Declare(
# pybind11
# GIT_REPOSITORY https://github.com/pybind/pybind11
# GIT_TAG v2.11.1
# )
# set(PYBIND11_FINDPYTHON CACHE INTERNAL false)
# set(PYBIND11_INSTALL CACHE INTERNAL false)
# set(PYBIND11_INTERNALS_VERSION CACHE INTERNAL "")
# set(PYBIND11_NOPYTHON CACHE INTERNAL false)
# set(PYBIND11_PYTHON_VERSION CACHE INTERNAL "")
# set(PYBIND11_TEST CACHE INTERNAL false)
# set(pybind11_DIR CACHE INTERNAL "")

# ----- Lua -----
FetchContent_Declare(lua
    GIT_REPOSITORY https://github.com/lua/lua.git
    GIT_TAG v5.4.6
    GIT_SHALLOW ON
    GIT_PROGRESS ON
)
FetchContent_GetProperties(lua)

if(NOT cpputest_POPULATED)
    FetchContent_Populate(lua)
endif()

# ----- Sol2 -----
FetchContent_Declare(sol2
    GIT_REPOSITORY https://github.com/ThePhD/sol2.git
    GIT_TAG v3.3.0
    GIT_SHALLOW ON
    GIT_PROGRESS ON
)

FetchContent_MakeAvailable(imgui eigen glfw sol2) # pybind11)

set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
message(STATUS "Added glfw in: " ${glfw_SOURCE_DIR})
set(glfw_INCLUDE_DIRS ${glfw_SOURCE_DIR}/include)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC . ${glfw_INCLUDE_DIRS} ${Vulkan_INCLUDE_DIRS} ${imgui_SOURCE_DIR})
target_link_directories(imgui PRIVATE ${CMAKE_BINARY_DIR})
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

set(CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL_BACKUP})
message(STATUS "Added dear imgui in: " ${imgui_SOURCE_DIR})

file(GLOB_RECURSE LUA_SOURCES "${lua_SOURCE_DIR}/*.c")
file(GLOB_RECURSE LUA_HEADER "${lua_SOURCE_DIR}/*.h")

add_library(lua STATIC ${LUA_SOURCES} ${LUA_HEADER})
target_include_directories(lua PUBLIC ${lua_SOURCE_DIR})
add_library(lua::lua ALIAS lua)
message(STATUS "Added Lua in: " ${lua_SOURCE_DIR})

message(STATUS "Added Sol2 in: " ${sol2_SOURCE_DIR})

unset(BUILD_TESTING CACHE)
message(STATUS "Added Eigen in: " ${eigen_SOURCE_DIR})

if(SAF_BUILD_DOC)
    find_package(Doxygen REQUIRED dot)

    if(DOXYGEN_FOUND)
        message(STATUS "Found Doxygen!")
        add_subdirectory(doc)
    else()
        message(WARNING "Doxygen not found, documentation will not be build. Please Install Doxygen to build the documentation!")
    endif()
endif()

message(STATUS "================================================")

set(SAF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/saf)

add_subdirectory(saf)

# add_subdirectory(pysaf)
add_subdirectory(examples)

message(STATUS "Done!")
message(STATUS "================================================")
