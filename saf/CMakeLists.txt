project(saf)

message(STATUS "================================================")
message(STATUS "Adding saf library!")

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(saf
    STATIC
    ${HEADER}
    ${SOURCES}
)

target_link_libraries(saf
    PUBLIC
    Vulkan::Vulkan
    Eigen3::Eigen
    imgui
    PRIVATE
    glfw
)

target_include_directories(saf
    PUBLIC
    ${SAF_INCLUDE_DIR}
)

set(SAF_PRECOMPILED_HEADER
    core/types.hpp
)

target_precompile_headers(saf
    PUBLIC
    ${SAF_PRECOMPILED_HEADER}
)

target_compile_definitions(saf
    PUBLIC
    $<$<CONFIG:Debug>:SAF_DEBUG>
    $<$<CXX_COMPILER_ID:MSVC>: _CRT_SECURE_NO_WARNINGS>
    PRIVATE
    $<$<BOOL:${WIN32}>:WIN32>
    $<$<BOOL:${LINUX}>:LINUX>
)

target_compile_options(saf
    PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>: /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-gnu-zero-variadic-macro-arguments -Wno-documentation-unknown-command>
)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
  target_link_libraries(saf PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(saf::saf ALIAS saf)

install(TARGETS saf DESTINATION lib)
install(DIRECTORY core/include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY vulkan/include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY app/include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")

message(STATUS "================================================")
