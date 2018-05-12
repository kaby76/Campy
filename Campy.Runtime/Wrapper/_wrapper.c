
#include "_BCL_.h"
#include "MetaData.h"
#include "System.Array.h"
#include "Type.h"
#include "Types.h"
#include "basics.h"
#include "Heap.h"
#include "CLIFile.h"

#ifdef __cplusplus
extern "C" {
#endif


	EXPORT void InitTheBcl(void * g, size_t size, size_t first_overhead, int count, void * s)
	{
		InternalInitTheBcl(g, size, first_overhead, count, s);
	}

	EXPORT void CheckHeap()
	{
		InternalCheckHeap();
	}

	EXPORT void InitFileSystem()
{
	InternalInitFileSystem();
}

EXPORT void GfsAddFile(void * name, void * file, size_t length, void * result)
{
	InternalGfsAddFile(name, file, length, result);
}

EXPORT void InitializeBCL1()
{
	InternalInitializeBCL1();
}

EXPORT void InitializeBCL2()
{
	InternalInitializeBCL2();
}

EXPORT void* BclHeapAlloc(void* type_def)
{
	void * result = (void*)Heap_AllocType((tMD_TypeDef *)type_def);
	return result;
}

EXPORT void* BclArrayAlloc(void* element_type_def, int rank, unsigned int* lengths)
{
	tMD_TypeDef* array_type_def = Type_GetArrayTypeDef((tMD_TypeDef*)element_type_def, NULL, NULL);
	return (void*)SystemArray_NewVector(array_type_def, rank, lengths);
}

EXPORT void* BclGetMetaOfType(char* assemblyName, char* nameSpace, char* name, void* nested)
{
	tMD_TypeDef* result = MetaData_GetTypeDefFromFullNameAndNestedType(assemblyName, nameSpace, name, (tMD_TypeDef*)nested);
	MetaData_Fill_TypeDef(result, NULL, NULL);
	return (void*)result;
}

EXPORT void GcCollect()
{
	Heap_GarbageCollect();
}


EXPORT void * STDCALL BclGetMeta(char * file_name)
{
	tMetaData* result = CLIFile_GetMetaDataForAssembly(file_name);
	return (void*)result;
}

EXPORT void STDCALL BclPrintMeta(void* meta)
{
	CLIFile_PrintMetaData((tMetaData*)meta);
}

#ifdef __cplusplus
}
#endif
