//#include "Compat.h"
#include "cuda.h"
#include <cstdarg>
#include "Gvsnprintf.h"

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
	unsigned char u1, u2;
	unsigned char * s1 = (unsigned char *)_Buf1;
	unsigned char * s2 = (unsigned char *)_Buf2;
	for (; _Size--; s1++, s1++) {
		u1 = *s1;
		u2 = *s2;
		if (u1 != u2) {
			return (u1 - u2);
		}
	}
	return 0;
}

__device__ void* __cdecl gpumemcpy(
	void* _Dst,
	void const* _Src,
	size_t      _Size
)
{
	return memcpy(_Dst, _Src, _Size);
}


__device__ int gpusprintf(
	char*       const _Buffer,
	char const* const _Format,
	...)
{
	va_list arg;
	int done;
	va_start(arg, _Format);
	done = Gvsprintf(_Buffer, _Format, arg);
	va_end(arg);
	return done;
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
	void * result = malloc(_Size);
	memcpy(result, _Block, _Size);
	free(_Block);
    return result;
}


__device__ void* gpumalloc(
	size_t _Size
)
{
	void * result = malloc(_Size);
    return result;
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
	return memset(_Dst, _Val, _Size);
}


__device__ char * gpustrcpy(char * destination, const char * source)
{
    return NULL;
}


