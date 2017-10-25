#include "Compat.h"

__device__ int gpustrcmp(
	char const* _Str1,
	char const* _Str2
)
{
    return 0;
}


__device__  void gpuexit(int _Code) {}

__device__ size_t gpustrlen(
	char const* _Str
)
{
    return 0;
}

__device__ int gpustrncmp(
	char const* _Str1,
	char const* _Str2,
	size_t      _MaxCount
)
{
    return 0;
}

__device__ int  gpumemcmp(
	void const* _Buf1,
	void const* _Buf2,
	size_t      _Size
)
{
    return 0;
}

__device__ void* __cdecl gpumemcpy(
	void* _Dst,
	void const* _Src,
	size_t      _Size
)
{
    return NULL;
}


__device__ int gpusprintf(
	char*       const _Buffer,
	char const* const _Format,
	...)
{
    return 0;
}


__device__ char*  gpustrchr(char* const _String, int const _Ch)
{
    return NULL;
}


__device__ void* __cdecl gpurealloc(
	void*  _Block,
	size_t _Size
)
{
    return NULL;
}


__device__ void* gpumalloc(
	size_t _Size
)
{
    return NULL;
}


__device__ char * gpustrcat(char * destination, const char * source)
{
    return NULL;
}



__device__ void* gpumemset(
	void*  _Dst,
	int    _Val,
	size_t _Size
)
{
    return NULL;
}


__device__ char * gpustrcpy(char * destination, const char * source)
{
    return NULL;
}


