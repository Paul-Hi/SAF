if(TARGET saf::imgui)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.92.8-docking
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(imgui)

if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot
    GIT_TAG 262de5114e562aba56e0cab719c31a61e798b47e
)

FetchContent_GetProperties(implot)

if(NOT implot_POPULATED)
    FetchContent_Populate(implot)
endif()

FetchContent_Declare(
    implot3d
    GIT_REPOSITORY https://github.com/brenocq/implot3d.git
    GIT_TAG 41ae3e447c0de20ecab95d38a4b4dc0835a3efc2
)

FetchContent_GetProperties(implot3d)

if(NOT implot3d_POPULATED)
    FetchContent_Populate(implot3d)
endif()

FetchContent_Declare(
    imfile
    GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
    GIT_TAG v0.6.8
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(imfile)

if(NOT imfile_POPULATED)
    FetchContent_Populate(imfile)
endif()

message(STATUS "Creating Target 'saf::imgui'")

add_library(saf_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d.cpp
    ${implot3d_SOURCE_DIR}/implot3d_demo.cpp
    ${implot3d_SOURCE_DIR}/implot3d_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d_meshes.cpp
    ${imfile_SOURCE_DIR}/ImGuiFileDialog.cpp
)
target_compile_definitions(saf_imgui PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)

add_library(saf::imgui ALIAS saf_imgui)

target_include_directories(saf_imgui PUBLIC . ${glfw_INCLUDE_DIRS} ${imgui_SOURCE_DIR} ${implot_SOURCE_DIR} ${implot3d_SOURCE_DIR} ${imfile_SOURCE_DIR})
target_link_directories(saf_imgui PRIVATE ${CMAKE_BINARY_DIR})
include(volk)
include(glfw)
target_link_libraries(saf_imgui PUBLIC saf::glfw saf::volk_headers)
