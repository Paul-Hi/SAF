project(template_cuda)

set(APP_NAME template_cuda)

message(STATUS "================================================")
message(STATUS "Adding template_cuda!")

if(WIN32 AND MSVC)
    add_executable(
        ${APP_NAME} WIN32
        template_cuda.cpp
        template_cuda.cu
    )
else()
    add_executable(
        ${APP_NAME}
        template_cuda.cpp
        template_cuda.cu
    )
endif()

set_target_properties(${APP_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/bin
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${APP_NAME}
    PUBLIC
    saf::saf
)

target_include_directories(${APP_NAME}
    PRIVATE
    ${SAF_INCLUDE_DIR}
)

target_compile_options(${APP_NAME}
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>: /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-gnu-zero-variadic-macro-arguments -Wno-global-constructors -Wno-documentation-unknown-command>
    # Fast math, as close to Vulkan shader math as CUDA gets: hardware sinf/cosf/expf/logf, approximate division and sqrt,
    # denormals flushed to zero. Remove it if higher precision is required; precise sinf makes the kernel about 2.5x slower.
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_compile_definitions(${APP_NAME}
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>: _CRT_SECURE_NO_WARNINGS>
)

if(WIN32)
    set(SUBSYSTEM_LINKER_OPTIONS "/SUBSYSTEM:CONSOLE")
endif()

target_link_options(${APP_NAME} PRIVATE ${SUBSYSTEM_LINKER_OPTIONS})

install(TARGETS ${APP_NAME} DESTINATION bin)

message(STATUS "================================================")
