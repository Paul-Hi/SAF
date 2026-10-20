# SAF

A Simple Application Framework with Vulkan and ImGui to quickly and easily create C/C++ applications with a graphical user interface.

## Use

After checking out the repository, CMake should automatically set up all dependencies except the Vulkan SDK.

SAF is not installed as a prebuilt library. Add it to your project's CMake build with `add_subdirectory` or `FetchContent`, link `saf::saf`, and add SAF's `saf/` directory to your include path (`SAF_INCLUDE_DIR`), as `examples/template_slang/CMakeLists.txt` does. `examples/template_slang` (Vulkan compute with Slang) and `examples/template_cuda` (CUDA interop) are minimal applications to copy as a starting point. They are built only when SAF is the top-level project, unless `SAF_BUILD_EXAMPLES` says otherwise.

`skills/saf/` is an agent skill for creating and writing SAF applications; its `scripts/new-project.sh` scaffolds a project from either template. To use it with Claude Code, link it into your skills: `ln -s "$PWD/skills/saf" ~/.claude/skills/saf`.

## Dependencies

- [Vulkan](https://www.vulkan.org)
- [volk](https://github.com/zeux/volk)
- [glfw](https://github.com/glfw/glfw)
- [ImGui](https://github.com/ocornut/imgui), [ImPlot](https://github.com/epezent/implot) , [ImPlot3d](https://github.com/brenocq/implot3d.git) and [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog)
- [Eigen](https://gitlab.com/libeigen/eigen)
- [SPIRV-Reflect](https://github.com/KhronosGroup/SPIRV-Reflect), with `SAF_ENABLE_VULKAN_COMPUTE`
- `glslc` from [shaderc](https://github.com/google/shaderc) and `slangc` from [Slang](https://github.com/shader-slang/slang), for `saf_add_shaders`. CMake does not fetch them; each is needed only for the shaders written in its language.
