if(TARGET saf::sol2)
    return()
endif()

if(TARGET sol2)
    message(STATUS "Target 'sol2' already exists, using it instead of creating a new one.")
    add_library(saf::sol2 ALIAS sol2)
    return()
endif()

include(lua)
include(FetchContent)

FetchContent_GetProperties(sol2)

if(NOT sol2_POPULATED)
    FetchContent_Declare(
        sol2
        GIT_REPOSITORY https://github.com/ThePhD/sol2.git
        GIT_TAG v3.3.0
        GIT_SHALLOW TRUE
    )
    set(SOL2_BUILD_LUA FALSE CACHE BOOL "Always build Lua, do not search for it in the system")
    FetchContent_MakeAvailable(sol2)
endif()

message(STATUS "Creating Target 'saf::sol2'")

add_library(saf::sol2 ALIAS sol2)
