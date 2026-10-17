#pragma once

#include <core/types.hpp>
#include <cuda_runtime.h>

void callVoronoiKernel(cudaSurfaceObject_t imageSurface, saf::U32 width, saf::U32 height, saf::F32 time, saf::F32 maxDistance, saf::U32 seed);
