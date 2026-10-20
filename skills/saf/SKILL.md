---
name: saf
description: Build GPU GUI applications on SAF (C++20, Vulkan + ImGui, Slang compute or CUDA interop). Use when creating a new SAF project, or writing code against SAF's Application, Layer, images, buffers, compute pipelines or CUDA interop in a project that consumes SAF.
---

# SAF applications

SAF is a static library: an **Application** owns the window, the Vulkan context, the frame loop and every GPU resource. User code lives in **Layers** pushed onto the Application. Resources are named by typed **Handles** (`ImageHandle`, `BufferHandle`, `ComputePipelineHandle`). A project pulls SAF in with CMake `FetchContent`, pinned to one commit.

## 1. Scaffold

Pick the backend before scaffolding:

- **Slang compute** (default): a Slang shader dispatched through SAF. Runs on any Vulkan GPU. Needs `slangc` on PATH.
- **CUDA** (`--cuda`): a CUDA kernel writing an image SAF shares with Vulkan. NVIDIA only. Needs the CUDA toolkit.

Run `<SAF checkout>/skills/saf/scripts/new-project.sh <directory> [--cuda] [--ref <git ref>]`. The directory's basename becomes the target and file name, so it must be an identifier. The script copies the matching template (an animated Voronoi field) into `src/`. It writes a `CMakeLists.txt` that fetches SAF at the checkout's `HEAD` unless `--ref` is given. Relay any warning about an unpushed or uncommitted SAF to the user.

Done when `<directory>/CMakeLists.txt` and `<directory>/src/` exist.

## 2. Build

```
cmake -S <dir> -B <dir>/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build <dir>/build
```

The first configure needs network: it fetches SAF and its dependencies. To build against a local SAF checkout instead, which you need while SAF itself is being changed, add `-DFETCHCONTENT_SOURCE_DIR_SAF=<SAF checkout>`. The Vulkan SDK, a C++20 compiler and Ninja must be installed; inside the SAF repo, `nix develop` provides them.

Done when the build exits 0 and `<dir>/build/release/bin/<name>` exists. The `release` directory is lowercase whatever the build type is spelled; Debug builds go to `debug/`. For Slang, `bin/shaders/<name>.slang.spv` must exist too.

## 3. Write the application

The SAF headers are the API reference: `<dir>/build/_deps/saf-src/saf/` after the first configure, or under `FETCHCONTENT_SOURCE_DIR_SAF`. Every public call is documented there. Read the header before you use a call:

| Need | Header |
|---|---|
| Application, settings, images, buffers, pipelines, dispatch, timing | `app/application.hpp` |
| Layer callbacks | `app/layer.hpp` |
| ImGui-editable values (`F32Parameter`, `Vec3Parameter`, …) | `app/parameter.hpp` |
| `VK_CHECK`, CUDA host/device macros | `core/helpers.hpp` |
| `U32`, `F32`, `Vec4` (Eigen), `Str` | `core/types.hpp` |
| GPU timing window, log window | `ui/gpuTimings.hpp`, `ui/log.hpp` |

Includes are rooted at `saf/`: `#include <app/application.hpp>`.

### Frame model

Each callback has one job. A compute or timing call outside `onUpdate`, and other API misuse, prints `[vulkan] Error: <what>` to stderr and exits.

- `onAttach`: create resources and pipelines, upload initial data, set bindings.
- `onUpdate(app, dt)`: record GPU work (`setPushConstants`, `dispatch`, CUDA launches, `beginTiming`/`endTiming`). SAF submits after `onUpdate` and makes the frame wait for that work before ImGui samples the result (frame-locked).
- `onUIRender`: ImGui only. `app->getTextureId(image)` shows an image through `ImGui::Image` or `ImPlot::PlotImage`.
- `onDetach`: destroy what `onAttach` created.

### Compute rules

- An image that compute writes is created with `.sharing = ImageSharing::Compute`; one that CUDA writes uses `ImageSharing::Cuda`. An image takes one or the other, never both.
- A binding keeps its image or buffer until rebound, so bind once in `onAttach`. Name a binding by its shader name (`"outImage"`) or by `{ set, binding }`.
- A push-constant struct matches the shader block byte for byte, padding included. A size mismatch aborts.
- Dispatches in one frame that read each other's writes need `app->computeBarrier()` between them. SAF orders work across frames itself.
- Each new shader is added to `saf_add_shaders(<target> SOURCES …)` in `CMakeLists.txt` and loaded with `loadSpirv(shaderPath("<file>.slang.spv"), words)`. A Slang entry point keeps its own name: pass `.entryPoint` when it is not `main`.
- Timing-scope names are unique across Vulkan and CUDA. `UIGpuTimings::render` shows them.

### Persisted settings

`Application(settings)` reads `saf.ini` from the working directory. Once that file exists, its name, window size, font scale, theme, clear color and V-Sync override the values in code. When a change to those fields shows no effect, delete `saf.ini`, or pass `false` as the second constructor argument while iterating.

## 4. Run

```
cd <dir>/build/release/bin && ./<name>
```

The app needs a Wayland or X11 session with a Vulkan GPU. Without one it exits with "Failed to initialize GLFW". When you have no display, finish at a clean build and hand the run to the user, naming what they should see.

## Updating SAF

Change `GIT_TAG` in `FetchContent_Declare(saf …)` to the new commit or tag, then reconfigure.
