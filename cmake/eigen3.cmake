if(TARGET saf::eigen3)
    return()
endif()

if(TARGET Eigen3::Eigen)
    message(STATUS "Target 'Eigen3::Eigen' already exists, using it instead of creating a new one.")
    add_library(saf::eigen3 ALIAS Eigen3::Eigen)
    return()
endif()

include(FetchContent)

FetchContent_GetProperties(eigen3)

if(NOT eigen3_POPULATED)
    FetchContent_Declare(
        eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_Populate(eigen3)
endif()

message(STATUS "Creating Target 'saf::eigen3'")

add_library(saf_eigen3 INTERFACE)
add_library(saf::eigen3 ALIAS saf_eigen3)

target_include_directories(saf_eigen3 INTERFACE ${eigen3_SOURCE_DIR})
