
#include "_BCL_.h"
#include "MetaData.h"
#include "MetaData_Search.h"
#include "System.Array.h"
#include "System.String.h"
#include "Type.h"
#include "Types.h"
#include "basics.h"
#include "Heap.h"
#include "CLIFile.h"
#include "Generics.h"

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

EXPORT void* BclGetArrayTypeDef(void* element_type_def, int rank)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclArrayAlloc\n");
	tMD_TypeDef* array_type_def = Type_GetArrayTypeDef((tMD_TypeDef*)element_type_def, rank, NULL, NULL);
	return (void*)array_type_def;
}

EXPORT void* BclArrayAlloc(void* element_type_def, int rank, unsigned int* lengths)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclArrayAlloc\n");
	tMD_TypeDef* array_type_def = Type_GetArrayTypeDef((tMD_TypeDef*)element_type_def, rank, NULL, NULL);
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

EXPORT void* BclGenericsGetGenericTypeFromCoreType(void * c, U32 numTypeArgs, void * a)
{
	tMD_TypeDef * pCoreType = (tMD_TypeDef *)c;
	tMD_TypeDef ** ppTypeArgs = (tMD_TypeDef **)a;
	tMD_TypeDef * result = Generics_GetGenericTypeFromCoreType(pCoreType, numTypeArgs, ppTypeArgs);
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
	tCLIFile* result = CLIFile_Load(file_name);
	return (void*)result;
}

EXPORT void STDCALL BclPrintMeta(void* meta)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclPrintMeta\n");
	if (meta == 0) return;
	tCLIFile* clifile = (tCLIFile*)meta;
	Gprintf("%s\n", clifile->pFileName);
	Gprintf("%s\n", clifile->pVersion);
	Gprintf("\n");
	MetaData_PrintMetaData(clifile->pMetaData);
}

EXPORT void * STDCALL BclAllocString(int len, void * chars)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("BclAllocString\n");
	return Internal_SystemString_FromCharPtrUTF16(len, (U16*)chars);
}

EXPORT void * BclHeapGetType(void * heapEntry)
{
	tMD_TypeDef* type = Heap_GetType((HEAP_PTR)heapEntry);
	return (void*)type;
}

EXPORT void * BclFindFieldInType(void * bcl_type, char * name)
{
	return (void *)MetaData_FindFieldInType((tMD_TypeDef *)bcl_type, name);
}

EXPORT void * BclGetField(void * bcl_object, void * bcl_field)
{
	return (void *)MetaData_GetField((HEAP_PTR)bcl_object, (tMD_FieldDef *)bcl_field);
}

EXPORT void BclSetField(void * bcl_object, void * bcl_field, void * value)
{
	MetaData_SetField((HEAP_PTR)bcl_object, (tMD_FieldDef *)bcl_field, (HEAP_PTR)value);
}

EXPORT void BclGetFields(void * bcl_type, void * out_buf, void * out_len)
{
	MetaData_GetFields((tMD_TypeDef*)bcl_type, (tMD_FieldDef ***)out_buf, (int*)out_len);
}

EXPORT char * BclGetFieldName(void * bcl_field)
{
	char * name = MetaData_GetFieldName((tMD_FieldDef*)bcl_field);
	return name;
}

EXPORT void * BclGetFieldType(void * bcl_field)
{
	tMD_TypeDef* bcl_type = MetaData_GetFieldType((tMD_FieldDef*)bcl_field);
	return (void *)bcl_type;
}

EXPORT int BclSystemArrayGetRank(void * bcl_object)
{
	return SystemArray_GetRank((HEAP_PTR) bcl_object);
}

EXPORT void * BclSystemArrayGetDims(void * bcl_object)
{
	return SystemArray_GetDims((HEAP_PTR) bcl_object);
}

EXPORT void BclSystemArrayLoadElementIndices(void * bcl_object, unsigned int dim, void * indices, void * value)
{
	SystemArray_LoadElementIndices((HEAP_PTR)bcl_object, dim, (U64*)indices, (U64*)value);
}

EXPORT void BclSystemArrayLoadElementIndicesAddress(void * bcl_object, unsigned int dim, void * indices, void * value_address)
{
	SystemArray_LoadElementIndicesAddress((HEAP_PTR)bcl_object, dim, (U64*)indices, (HEAP_PTR*)value_address);
}

#ifdef __cplusplus
}
#endif
