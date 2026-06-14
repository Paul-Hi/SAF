if(TARGET saf::volk_headers)
    return()
endif()

if(TARGET volk_headers)
    message(STATUS "Target 'volk_headers' already exists, using it instead of creating a new one.")
    add_library(saf::volk_headers ALIAS volk_headers)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    volk_headers
    GIT_REPOSITORY https://github.com/zeux/volk
    GIT_TAG 1.4.350
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(volk_headers)

if(NOT volk_headers_POPULATED)
    set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
    set(VOLK_HEADERS_ONLY ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(volk_headers)
endif()

message(STATUS "Creating Target 'saf::volk_headers'")

add_library(saf::volk_headers ALIAS volk_headers)
