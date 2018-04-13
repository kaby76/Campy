
__device__  __host__ void __cdecl InternalInitTheBcl(void * g, size_t size, int count, void * s);

__declspec(dllexport) void InitTheBcl(void * g, size_t size, int count, void * s)
{
	InternalInitTheBcl(g, size, count, s);
}

__device__ __host__ void InternalInitFileSystem();

__declspec(dllexport) void InitFileSystem()
{
	InternalInitFileSystem();
}

__device__ __host__ void InternalGfsAddFile(void * name, void * file, size_t length, void * result);

__declspec(dllexport) void GfsAddFile(void * name, void * file, size_t length, void * result)
{
	InternalGfsAddFile(name, file, length, result);
}

__device__ __host__ void InternalInitializeBCL1();

__declspec(dllexport) void InitializeBCL1()
{
	InternalInitializeBCL1();
}

__device__ __host__ void InternalInitializeBCL2();

__declspec(dllexport) void InitializeBCL2()
{
	InternalInitializeBCL2();
}

__device__ __host__ void* Bcl_Heap_Alloc(char* assemblyName, char* nameSpace, char* name);

__declspec(dllexport) void* BclHeapAlloc(char* assemblyName, char* nameSpace, char* name)
{
	return Bcl_Heap_Alloc(assemblyName, nameSpace, name);
}

__device__ __host__ void* Bcl_Array_Alloc(char* assemblyName, char* nameSpace, char* name, int length);

__declspec(dllexport) void* BclArrayAlloc(char* assemblyName, char* nameSpace, char* name, int length)
{
	return Bcl_Array_Alloc(assemblyName, nameSpace, name, length);
}
