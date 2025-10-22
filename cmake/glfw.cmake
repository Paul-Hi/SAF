if(TARGET saf::glfw)
    return()
endif()

if(TARGET glfw)
    message(STATUS "Target 'glfw' already exists, using it instead of creating a new one.")
    add_library(saf::glfw ALIAS glfw)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(glfw)

if(NOT glfw_POPULATED)
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
    set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
    set(glfw_INCLUDE_DIRS ${glfw_SOURCE_DIR}/include)

    FetchContent_MakeAvailable(glfw)
endif()

message(STATUS "Creating Target 'saf::glfw'")

add_library(saf::glfw ALIAS glfw)
