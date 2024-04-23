project(demo)

set(APP_NAME demo)

message(STATUS "================================================")
message(STATUS "Adding demo!")

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if(WIN32 AND MSVC)
    add_executable(
        ${APP_NAME} WIN32
        ${SOURCES}
    )
else()
    add_executable(
        ${APP_NAME}
        ${SOURCES}
    )
endif()

set_target_properties(${APP_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}$<$<CONFIG:Debug>:/debug>$<$<CONFIG:Release>:/release>/bin
)

target_link_libraries(${APP_NAME}
    PUBLIC
    saf::saf
)

target_include_directories(${APP_NAME}
    PRIVATE
    ${SAF_INCLUDE_DIR}
)

if(SAF_ENABLE_CUDA_INTEROP)
    set_target_properties(${APP_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_compile_options(${APP_NAME}
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>: /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-gnu-zero-variadic-macro-arguments -Wno-global-constructors -Wno-documentation-unknown-command>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v -Xcompiler=-fopenmp>
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
install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets/ DESTINATION bin/assets)
install(FILES ${CMAKE_SOURCE_DIR}/imgui.ini DESTINATION bin)

message(STATUS "================================================")
