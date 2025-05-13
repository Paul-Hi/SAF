if(TARGET saf::efsw)
    return()
endif()

if(TARGET efsw)
    message(STATUS "Target 'efsw' already exists, using it instead of creating a new one.")
    add_library(saf::efsw ALIAS efsw)
    return()
endif()

include(lua)
include(FetchContent)

FetchContent_GetProperties(efsw)

if(NOT efsw_POPULATED)
    FetchContent_Declare(
        efsw
        GIT_REPOSITORY https://github.com/SpartanJ/efsw
        GIT_TAG 1.4.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(efsw)
endif()

message(STATUS "Creating Target 'saf::efsw'")

add_library(saf::efsw ALIAS efsw)
