#!/usr/bin/env bash
# Creates a SAF application from one of SAF's example templates.
# Usage: new-project.sh <directory> [--cuda] [--ref <git ref>]
set -euo pipefail

usage() {
    echo "usage: $0 <directory> [--cuda] [--ref <git ref>]" >&2
    exit 2
}

dir=""
cuda=false
ref=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) cuda=true ;;
        --ref) [[ $# -ge 2 ]] || usage; ref="$2"; shift ;;
        -*) usage ;;
        *) [[ -z "$dir" ]] || usage; dir="$1" ;;
    esac
    shift
done
[[ -n "$dir" ]] || usage

name="$(basename "$dir")"
if [[ ! "$name" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]; then
    echo "error: '$name' is not a valid project name: use letters, digits and _, starting with a letter" >&2
    exit 1
fi
if [[ -e "$dir" ]]; then
    echo "error: $dir already exists" >&2
    exit 1
fi

saf="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Pins the SAF commit this script was taken from, which is the version the skill describes
if [[ -z "$ref" ]]; then
    ref="$(git -C "$saf" rev-parse HEAD)"
    if [[ -z "$(git -C "$saf" branch -r --contains "$ref" 2>/dev/null)" ]]; then
        echo "warning: SAF commit $ref is not on any remote branch; fetching it fails until it is pushed." >&2
        echo "         Configure with -DFETCHCONTENT_SOURCE_DIR_SAF=$saf meanwhile." >&2
    fi
    if [[ -n "$(git -C "$saf" status --porcelain -- saf cmake CMakeLists.txt)" ]]; then
        echo "warning: SAF has uncommitted library changes that the pinned commit $ref lacks." >&2
    fi
fi

if $cuda; then
    template=template_cuda
    sources=(template_cuda.cpp template_cuda.cu template_cuda.cuh)
else
    template=template_slang
    sources=(template_slang.cpp shaders/template_slang.slang)
fi

mkdir -p "$dir/src"
for source in "${sources[@]}"; do
    target="$dir/src/${source//$template/$name}"
    mkdir -p "$(dirname "$target")"
    sed -e "s/$template/$name/g" -e "s/\"Template Slang\"/\"$name\"/" -e "s/\"Template CUDA\"/\"$name\"/" \
        "$saf/examples/$template/$source" > "$target"
done
cp "$saf/.clang-format" "$dir/"

if $cuda; then
    languages="CXX C CUDA"
    options=$'set(SAF_ENABLE_CUDA_INTEROP ON)\n# RTX 50 series, as SAF defaults to; SAF and the application must match\nset(CMAKE_CUDA_ARCHITECTURES 120)'
    executable="src/$name.cpp src/$name.cu"
    properties=$'\n    CUDA_SEPARABLE_COMPILATION ON'
    # The template's kernel is tuned for fast math; remove it if higher precision is required
    compileOptions=$'\n    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>'
    shaders=""
else
    languages="CXX C"
    options=""
    executable="src/$name.cpp"
    properties=""
    compileOptions=""
    shaders=$'\n# Puts the SPIR-V in shaders/ next to the executable, where shaderPath() looks\nsaf_add_shaders('"$name"$'\n    SOURCES\n    src/shaders/'"$name"$'.slang\n)\n'
fi

cat > "$dir/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.26)

project($name LANGUAGES $languages)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
$options

include(FetchContent)
FetchContent_Declare(saf
    GIT_REPOSITORY https://github.com/Paul-Hi/SAF.git
    GIT_TAG $ref
)
FetchContent_MakeAvailable(saf)

add_executable($name $executable)

set_target_properties($name
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}\$<\$<CONFIG:Debug>:/debug>\$<\$<CONFIG:Release>:/release>/bin$properties
)
$shaders
target_link_libraries($name PRIVATE saf::saf)

target_compile_options($name
    PRIVATE
    \$<\$<CXX_COMPILER_ID:MSVC>: /W4>
    \$<\$<NOT:\$<CXX_COMPILER_ID:MSVC>>: -Wall -Wextra>$compileOptions
)
EOF

echo "Created $dir from $template, SAF pinned at $ref."
echo "Build: cmake -S $dir -B $dir/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build $dir/build"
echo "Run:   cd $dir/build/release/bin && ./$name"
