#pragma once

#include "Types.h"

__device__ __host__ void InternalGfsAddFile(void * name, void * file, size_t length, void * result);
__device__  __host__ void InternalInitTheBcl(void * g, size_t size, size_t first_overhead, int count, void * s);
__device__ __host__ void InternalInitFileSystem();
__device__ __host__ void InternalInitializeBCL1();
__device__ __host__ void InternalInitializeBCL2();
__device__ __host__ void InternalCheckHeap();
__device__ __host__ void InternalSetOptions(U64 options);

