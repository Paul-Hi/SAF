project(saf)

message(STATUS "================================================")
message(STATUS "Adding saf library!")

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(saf
    STATIC
    ${SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cu
)

target_link_libraries(saf
    PUBLIC
    Vulkan::Vulkan
    saf::eigen3
    saf::imgui
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
    $<$<BOOL:${SAF_ENABLE_SCRIPTING}>:saf::lua>
    $<$<BOOL:${SAF_ENABLE_FILE_WATCH}>:saf::efsw>
    $<$<BOOL:${SAF_ENABLE_CUDA_INTEROP}>:${CUDA_LIBRARIES}>
    PRIVATE
    saf::glfw
)

target_include_directories(saf
    PUBLIC
    ${SAF_INCLUDE_DIR}
    $<$<BOOL:${SAF_ENABLE_SCRIPTING}>:${sol2_SOURCE_DIR}/include>
    $<$<BOOL:${SAF_ENABLE_FILE_WATCH}>:${efsw_SOURCE_DIR}/include>
    $<$<BOOL:${SAF_ENABLE_CUDA_INTEROP}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

if(SAF_ENABLE_CUDA_INTEROP)
    set_target_properties(saf
        PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

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
    $<$<BOOL:${SAF_ENABLE_CUDA_INTEROP}>:SAF_CUDA_INTEROP>
    PRIVATE
    $<$<BOOL:${WIN32}>:WIN32>
    $<$<BOOL:${LINUX}>:LINUX>
)

target_compile_options(saf
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>: /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-gnu-zero-variadic-macro-arguments -Wno-global-constructors -Wno-documentation-unknown-command>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v -Xcompiler=-fopenmp>
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
