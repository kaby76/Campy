#include "Compat.h"
#include "Sys.h"
#include "MetaData.h"
#include "JIT.h"
#include "Type.h"
#include "Finalizer.h"
#include "System.Net.Sockets.Socket.h"
#include "Gprintf.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

__device__  void gpuexit(int _Code) {}

__device__ size_t Gstrlen(
	char const* _Str
)
{
    size_t r = 0;
    for (size_t l = 0; _Str[l] != 0; r = ++l)
        ;
    return r;
}

//__device__ int  gpumemcmp(
//	void const* _Buf1,
//	void const* _Buf2,
//	size_t      _Size
//)
//{
//	unsigned char u1, u2;
//	unsigned char * s1 = (unsigned char *)_Buf1;
//	unsigned char * s2 = (unsigned char *)_Buf2;
//	for (; _Size--; s1++, s1++) {
//		u1 = *s1;
//		u2 = *s2;
//		if (u1 != u2) {
//			return (u1 - u2);
//		}
//	}
//	return 0;
//}

//__device__ void* __cdecl gpumemcpy(
//	void* _Dst,
//	void const* _Src,
//	size_t      _Size
//)
//{
//	return memcpy(_Dst, _Src, _Size);
//}




__device__ void* __cdecl Grealloc(
	void*  _Block,
	size_t _Size
)
{
	void * result = malloc(_Size);
	memcpy(result, _Block, _Size);
	free(_Block);
    return result;
}


__device__ void* Gmalloc(
	size_t _Size
)
{
	void * result = malloc(_Size);
    return result;
}



//__device__ void* Gmemset(
//	void*  _Dst,
//	int    _Val,
//	size_t _Size
//)
//{
//	return memset(_Dst, _Val, _Size);
//}


__global__
void Initialize_BCL1()
{
	//JIT_Execute_Init();
	MetaData_Init();
	//Heap_Init();
	//Finalizer_Init();
	//Socket_Init();
}

__global__
void Initialize_BCL2()
{
	//JIT_Execute_Init();
	//MetaData_Init();
	Type_Init();
	Heap_Init();
	Finalizer_Init();
	//Socket_Init();
}


__device__
void* Bcl_Heap_Alloc(STRING assemblyName, STRING nameSpace, STRING name)
{
	tMD_TypeDef* type_def = MetaData_GetTypeDefFromFullName(assemblyName, nameSpace, name);
	void * result = Heap_AllocType(type_def);
	return result;
}