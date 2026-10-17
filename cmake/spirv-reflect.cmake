if(TARGET saf::spirv-reflect-static)
    return()
endif()

if(TARGET spirv-reflect-static)
    message(STATUS "Target 'spirv-reflect-static' already exists, using it instead of creating a new one.")
    add_library(saf::spirv-reflect-static ALIAS spirv-reflect-static)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    spirv_reflect
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Reflect
    GIT_TAG vulkan-sdk-1.4.350.0 # NOTE This would ideally always be the same version as the volk we load!
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(spirv_reflect)

if(NOT spirv_reflect_POPULATED)
    set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)
    set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "" FORCE)
    set(SPIRV_REFLECT_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(spirv_reflect)
endif()

message(STATUS "Creating Target 'saf::spirv-reflect-static'")

add_library(saf::spirv-reflect-static ALIAS spirv-reflect-static)
