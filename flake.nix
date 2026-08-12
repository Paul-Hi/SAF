{
  description = "SAF development shell";

  # === Binary cache for pre-built CUDA packages ===
  # Nix will prompt to trust this substituter on first use.
  nixConfig = {
    extra-substituters = ["https://cache.nixos-cuda.org"];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    lib = nixpkgs.lib;

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell.override {
      stdenv = pkgs.overrideCC pkgs.stdenv pkgs.gcc13;
    } {

      packages = [
        # C++ Toolchain
        pkgs.gcc13
        pkgs.cmake
        pkgs.ninja
        pkgs.git
        pkgs.pkg-config
        pkgs.ccache

        # CUDA
        pkgs.linuxPackages.nvidia_x11
        pkgs.cudaPackages.cudatoolkit
        pkgs.cudaPackages.cccl

        # Vulkan
        pkgs.vulkan-loader
        pkgs.vulkan-headers
        pkgs.vulkan-validation-layers
        pkgs.vulkan-utility-libraries
        pkgs.vulkan-tools
        pkgs.shaderc
        pkgs.spirv-tools

        # Wayland + X11 (for GLFW)
        pkgs.wayland
        pkgs.wayland-scanner
        pkgs.wayland-protocols
        pkgs.libxkbcommon
        pkgs.libx11
        pkgs.libxrandr
        pkgs.libxinerama
        pkgs.libxcursor
        pkgs.libxi
      ];

      shellHook = ''
        # CUDA
        export CUDAHOSTCXX="${pkgs.gcc13}/bin/g++"

        # Thrust
        export Thrust_DIR="${pkgs.cudaPackages.cccl}/lib/cmake/thrust"

        # ccache
        export CCACHE_DIR="$HOME/.cache/ccache"
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

        # Runtime libraries
        # This ensures the dynamic linker can find stuff
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
            pkgs.wayland
            pkgs.libx11
            pkgs.libxkbcommon
            pkgs.vulkan-loader
            pkgs.cudaPackages.cudatoolkit
            pkgs.linuxPackages.nvidia_x11
          ]}:$LD_LIBRARY_PATH"

        echo ""
        echo "  DifferentiableRendering dev shell"
        echo "  CUDA ${pkgs.cudaPackages.cudatoolkit.version} | GCC $(gcc --version | head -1 | cut -d' ' -f3)"
        echo "  Vulkan headers: ${pkgs.vulkan-headers.version}"
        echo ""
        echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
      '';
    };
  };
}
