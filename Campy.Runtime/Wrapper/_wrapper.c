
#include "_BCL_.h"
#include "MetaData.h"
#include "System.Array.h"
#include "System.String.h"
#include "Type.h"
#include "Types.h"
#include "basics.h"
#include "Heap.h"
#include "CLIFile.h"

#ifdef __cplusplus
extern "C" {
#endif


	EXPORT void InitTheBcl(void * g, size_t size, size_t first_overhead, int count)
	{
		if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
			Gprintf("InitTheBcl\n");

		InternalInitTheBcl(g, size, first_overhead, count);
	}

	EXPORT void CheckHeap()
	{
		if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
			Gprintf("CheckHeap\n");

		InternalCheckHeap();
	}

	EXPORT void SetOptions(U64 options)
	{
		if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
			Gprintf("SetOptions\n");

		InternalSetOptions(options);
	}


	EXPORT void InitFileSystem()
{
	InternalInitFileSystem();
}

EXPORT void GfsAddFile(void * name, void * file, size_t length, void * result)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("GfsAddFile\n");

	//printf("Adding File to GFS %s 0x%08llx %x\n",
	//	name, file, length);
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
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclHeapAlloc\n");
	void * result = (void*)Heap_AllocType((tMD_TypeDef *)type_def);
	return result;
}

EXPORT void* BclArrayAlloc(void* element_type_def, int rank, unsigned int* lengths)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclArrayAlloc\n");
	tMD_TypeDef* array_type_def = Type_GetArrayTypeDef((tMD_TypeDef*)element_type_def, NULL, NULL);
	return (void*)SystemArray_NewVector(array_type_def, rank, lengths);
}

EXPORT void* BclGetMetaOfType(char* assemblyName, char* nameSpace, char* name, void* nested)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclGetMetaOfType\n");
	tMD_TypeDef* result = MetaData_GetTypeDefFromFullNameAndNestedType(assemblyName, nameSpace, name, (tMD_TypeDef*)nested);
	MetaData_Fill_TypeDef(result, NULL, NULL);
	return (void*)result;
}

EXPORT void GcCollect()
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("GcCollect\n");
	Heap_GarbageCollect();
}


EXPORT void * STDCALL BclGetMeta(char * file_name)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclGetMeta\n");
	tMetaData* result = CLIFile_GetMetaDataForAssembly(file_name);
	return (void*)result;
}

EXPORT void STDCALL BclPrintMeta(void* meta)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclPrintMeta\n");
	MetaData_PrintMetaData((tMetaData*)meta);
}

EXPORT void * STDCALL BclAllocString(int len, void * chars)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclAllocString\n");
	return Internal_SystemString_FromCharPtrUTF16(len, (U16*)chars);
}

#ifdef __cplusplus
}
#endif
