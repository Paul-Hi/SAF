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
    $<$<BOOL:${SAF_ENABLE_SCRIPTING}>:lua::lua>
    $<$<BOOL:${SAF_ENABLE_FILE_WATCH}>:efsw::efsw>
    PRIVATE
    glfw
)

target_include_directories(saf
    PUBLIC
    ${SAF_INCLUDE_DIR}
    $<$<BOOL:${SAF_ENABLE_SCRIPTING}>:${sol2_SOURCE_DIR}/include>
    $<$<BOOL:${SAF_ENABLE_FILE_WATCH}>:${efsw_SOURCE_DIR}/include>
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
    $<$<BOOL:${SAF_ENABLE_SCRIPTING}>:SAF_SCRIPTING>
    $<$<BOOL:${SAF_ENABLE_FILE_WATCH}>:SAF_FILE_WATCH>
    PRIVATE
    $<$<BOOL:${WIN32}>:WIN32>
    $<$<BOOL:${LINUX}>:LINUX>
)

target_compile_options(saf
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>: /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-gnu-zero-variadic-macro-arguments -Wno-global-constructors -Wno-documentation-unknown-command>
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
