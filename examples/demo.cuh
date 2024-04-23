#pragma once

#ifdef SAF_CUDA_INTEROP
#include <core/types.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

void callGrayScaleKernel(cudaSurfaceObject_t imageSurface, cudaTextureObject_t imageTexture, saf::I32 w, saf::I32 h);

#endif