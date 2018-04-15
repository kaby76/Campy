#pragma once
__device__ __host__ void InternalGfsAddFile(void * name, void * file, size_t length, void * result);
__device__  __host__ void __cdecl InternalInitTheBcl(void * g, size_t size, int count, void * s);
__device__ __host__ void InternalInitFileSystem();
__device__ __host__ void InternalInitializeBCL1();
__device__ __host__ void InternalInitializeBCL2();

