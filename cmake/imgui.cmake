if(TARGET saf::imgui)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.91.9b-docking
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(imgui)

if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot
    GIT_TAG 3da8bd34299965d3b0ab124df743fe3e076fa222
)

FetchContent_GetProperties(implot)

if(NOT implot_POPULATED)
    FetchContent_Populate(implot)
endif()

FetchContent_Declare(
    implot3d
    GIT_REPOSITORY https://github.com/brenocq/implot3d.git
    GIT_TAG v0.2
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(implot3d)

if(NOT implot3d_POPULATED)
    FetchContent_Populate(implot3d)
endif()

FetchContent_Declare(
    imfile
    GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
    GIT_TAG v0.6.7
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
add_library(saf::imgui ALIAS saf_imgui)

target_include_directories(saf_imgui PUBLIC . ${glfw_INCLUDE_DIRS} ${Vulkan_INCLUDE_DIRS} ${imgui_SOURCE_DIR} ${implot_SOURCE_DIR} ${implot3d_SOURCE_DIR} ${imfile_SOURCE_DIR})
target_link_directories(saf_imgui PRIVATE ${CMAKE_BINARY_DIR})
include(glfw)
target_link_libraries(saf_imgui PUBLIC saf::glfw Vulkan::Vulkan)
