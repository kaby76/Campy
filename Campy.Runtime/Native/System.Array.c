// Copyright (c) 2012 DotNetAnywhere
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "Compat.h"
#include "Sys.h"

#include "System.Array.h"

#include "Types.h"
#include "MetaData.h"
#include "Heap.h"
#include "Type.h"

// This structure must be kept consistent with code in Campy.Compiler, Campy.Runtime.Corlib, as well as this module.
typedef struct tSystemArray_ tSystemArray;

struct tSystemArray_ {
    PTR ptr_elements;
    U64 rank;
    // How many elements in array
    //U32[] lengths;
    // The elements
    //U8 elements[0];
};

function_space_specifier U32 SystemArray_Length(void * p)
{
	tSystemArray *pArray = (tSystemArray*)p;
	tMD_TypeDef *pArrayType;
	U32 index, elementSize;
	tMD_TypeDef *pElementType;
	PTR pElement;
	pArrayType = Heap_GetType((HEAP_PTR)p);
	U64* p_len = &pArray->rank;
	p_len++;
	U64 len = 1;
	for (int i = 0; i < pArray->rank; ++i)
		len = len * (*p_len);
	return (U32)len;
}

function_space_specifier tAsyncCall* System_Array_Internal_GetLength(PTR pThis_, PTR pParams, PTR pReturnValue) {
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayType;
    U32 index, elementSize;
    tMD_TypeDef *pElementType;
    PTR pElement;

    pArrayType = Heap_GetType(pThis_);
    //pElementType = pArrayType->pArrayElementType;
    //elementSize = pElementType->arrayElementSize;
    U64* p_len = &pArray->rank;
    p_len++;
    U64 len = 1;
    for (int i = 0; i < pArray->rank; ++i)
        len = len * (*p_len);

    *(U32*)pReturnValue = (U32)len;

    return NULL;
}

// Must return a boxed version of value-types
function_space_specifier tAsyncCall* System_Array_Internal_GetValue(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayType;
    U32 index, elementSize;
    tMD_TypeDef *pElementType;
    PTR pElement;

    index = *(U32*)pParams;
    pArrayType = Heap_GetType(pThis_);
    pElementType = pArrayType->pArrayElementType;
    elementSize = pElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    pElement = beginning_of_elements + elementSize * index;
    if (pElementType->isValueType) {
        // If it's a value-type, then box it
        HEAP_PTR boxedValue;
        if (pElementType->pGenericDefinition == _bcl_->types[TYPE_SYSTEM_NULLABLE]) {
            // Nullable type, so box specially
            if (*(U32*)pElement) {
                // Nullable has value
                boxedValue = Heap_AllocType(pElementType->ppClassTypeArgs[0]);
                // Don't copy the .hasValue part
                memcpy(boxedValue, pElement + 4, elementSize - 4);
            } else {
                // Nullable does not have value
                boxedValue = NULL;
            }
        } else {
            boxedValue = Heap_AllocType(pElementType);
            memcpy(boxedValue, pElement, elementSize);
        }
        *(HEAP_PTR*)pReturnValue = boxedValue;
    } else {
        // This must be a reference type, so it must be 32-bits wide
        *(U32*)pReturnValue = *(U32*)pElement;
    }

    return NULL;
}

// Value-types will be boxed
function_space_specifier tAsyncCall* System_Array_Internal_SetValue(PTR pThis_, PTR pParams, PTR pReturnValue) {
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayType, *pObjType;
    U32 index, elementSize;
    HEAP_PTR obj;
    tMD_TypeDef *pElementType;
    PTR pElement;

    pArrayType = Heap_GetType(pThis_);
    void **p = (void**)pParams;
    obj = *(HEAP_PTR*)p++;
    pObjType = Heap_GetType(obj);
    pElementType = pArrayType->pArrayElementType;
    // Check to see if the Type is ok to put in the array
    if (!(Type_IsAssignableFrom(pElementType, pObjType) ||
        (pElementType->pGenericDefinition == _bcl_->types[TYPE_SYSTEM_NULLABLE] &&
        pElementType->ppClassTypeArgs[0] == pObjType))) {
        // Can't be done
        *(U32*)pReturnValue = 0;
        return NULL;
    }

    index = *(U32*)p++;

#if defined(_MSC_VER) && defined(_DEBUG)
    // Do a bounds-check
    U32 len = *((&(pArray->rank)) + 1);
    if (index >= len) {
//      printf("[Array] Internal_SetValue() Bounds-check failed\n");
        __debugbreak();
    }
#endif

    elementSize = pElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    pElement = beginning_of_elements + elementSize * index;
    if (pElementType->isValueType) {
        if (pElementType->pGenericDefinition == _bcl_->types[TYPE_SYSTEM_NULLABLE]) {
            // Nullable type, so treat specially
            if (obj == NULL) {
                memset(pElement, 0, elementSize);
            } else {
                *(U32*)pElement = 1;
                memcpy(pElement + 4, obj, elementSize - 4);
            }
        } else {
            // Get the value out of the box
            memcpy(pElement, obj, elementSize);
        }
    } else {
        // This must be a reference type, so it must be 32-bits wide
        *(HEAP_PTR*)pElement = obj;
    }
    *(U32*)pReturnValue = 1;

    return NULL;
}

// Must return a boxed version of value-types
function_space_specifier tAsyncCall* System_Array_GetValue(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayType;
    U32 index, elementSize;
    tMD_TypeDef *pElementType;
    PTR pElement;

    index = *(U32*)pParams;
    pArrayType = Heap_GetType(pThis_);
    pElementType = pArrayType->pArrayElementType;
    elementSize = pElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    pElement = beginning_of_elements + elementSize * index;
    if (pElementType->isValueType) {
        // If it's a value-type, then box it
        HEAP_PTR boxedValue;
        if (pElementType->pGenericDefinition == _bcl_->types[TYPE_SYSTEM_NULLABLE]) {
            // Nullable type, so box specially
            if (*(U32*)pElement) {
                // Nullable has value
                boxedValue = Heap_AllocType(pElementType->ppClassTypeArgs[0]);
                // Don't copy the .hasValue part
                memcpy(boxedValue, pElement + 4, elementSize - 4);
            } else {
                // Nullable does not have value
                boxedValue = NULL;
            }
        } else {
            boxedValue = Heap_AllocType(pElementType);
            memcpy(boxedValue, pElement, elementSize);
        }
        *(HEAP_PTR*)pReturnValue = boxedValue;
    } else {
        // This must be a reference type, so it must be 32-bits wide
        *(U32*)pReturnValue = *(U32*)pElement;
    }

    return NULL;
}

// Value-types will be boxed
function_space_specifier tAsyncCall* System_Array_SetValue(PTR pThis_, PTR pParams, PTR pReturnValue) {
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayType, *pObjType;
    U32 index, elementSize;
    HEAP_PTR obj;
    tMD_TypeDef *pElementType;
    PTR pElement;

    pArrayType = Heap_GetType(pThis_);
    void **p = (void**)pParams;
    obj = *(HEAP_PTR*)p++;
    pObjType = Heap_GetType(obj);
    pElementType = pArrayType->pArrayElementType;
    // Check to see if the Type is ok to put in the array
    if (!(Type_IsAssignableFrom(pElementType, pObjType) ||
          (pElementType->pGenericDefinition == _bcl_->types[TYPE_SYSTEM_NULLABLE] &&
           pElementType->ppClassTypeArgs[0] == pObjType))) {
        // Can't be done
        *(U32*)pReturnValue = 0;
        return NULL;
    }

    index = *(U32*)p++;

#if defined(_MSC_VER) && defined(_DEBUG)
    // Do a bounds-check
    U32 len = *((&(pArray->rank)) + 1);
    if (index >= len) {
//      printf("[Array] Internal_SetValue() Bounds-check failed\n");
        __debugbreak();
    }
#endif

    elementSize = pElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    pElement = beginning_of_elements + elementSize * index;
    if (pElementType->isValueType) {
        if (pElementType->pGenericDefinition == _bcl_->types[TYPE_SYSTEM_NULLABLE]) {
            // Nullable type, so treat specially
            if (obj == NULL) {
                memset(pElement, 0, elementSize);
            } else {
                *(U32*)pElement = 1;
                memcpy(pElement + 4, obj, elementSize - 4);
            }
        } else {
            // Get the value out of the box
            memcpy(pElement, obj, elementSize);
        }
    } else {
        // This must be a reference type, so it must be 32-bits wide
        *(HEAP_PTR*)pElement = obj;
    }
    *(U32*)pReturnValue = 1;

    return NULL;
}

function_space_specifier tAsyncCall* System_Array_Clear(PTR pThis_, PTR pParams, PTR pReturnValue) {
    tSystemArray *pArray;
    U32 index, length, elementSize;
    tMD_TypeDef *pArrayType;

    void **p = (void**)pParams;
    pArray = *(tSystemArray**)p++;
    index = *(U32*)p++;
    length = *(U32*)p++;
    pArrayType = Heap_GetType((HEAP_PTR)pArray);
    elementSize = pArrayType->pArrayElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    memset(beginning_of_elements + index * elementSize, 0, length * elementSize);

    return NULL;
}

function_space_specifier tAsyncCall* System_Array_Internal_Copy(PTR pThis_, PTR pParams, PTR pReturnValue) {
    tSystemArray *pSrc, *pDst;
    tMD_TypeDef *pSrcType, *pDstType, *pSrcElementType;
    U32 srcIndex, dstIndex, length, elementSize;

    void ** p = (void**)pParams;
    pSrc = *(tSystemArray**)p++;
    srcIndex = *(U32*)p++;
    pDst = *(tSystemArray**)p++;
    dstIndex = *(U32*)p++;
    length = *(U32*)p++;
    
    // Check if we can do a fast-copy with these two arrays
    pSrcType = Heap_GetType((HEAP_PTR)pSrc);
    pDstType = Heap_GetType((HEAP_PTR)pDst);
    pSrcElementType = pSrcType->pArrayElementType;
    if (Type_IsAssignableFrom(pDstType->pArrayElementType, pSrcElementType)) {
        // Can do fast-copy

#if defined(_MSC_VER) && defined(_DEBUG)
        // Do bounds check
        U32 slen = *((&(pSrc->rank)) + 1);
        U32 dlen = *((&(pDst->rank)) + 1);
        if (srcIndex + length > slen || dstIndex + length > dlen) {
            //printf("[Array] Internal_Copy() Bounds check failed\n");
            __debugbreak();
        }
#endif

        elementSize = pSrcElementType->arrayElementSize;

        PTR beginning_of_elements_dst = pDst->ptr_elements;
        PTR beginning_of_elements_src = pSrc->ptr_elements;
        memcpy(beginning_of_elements_dst + dstIndex * elementSize, beginning_of_elements_src + srcIndex * elementSize, length * elementSize);

        *(U32*)pReturnValue = 1;
    } else {
        // Cannot do fast-copy
        *(U32*)pReturnValue = 0;
    }

    return NULL;
}

function_space_specifier tAsyncCall* System_Array_Resize(PTR pThis_, PTR pParams, PTR pReturnValue) {
    HEAP_PTR* ppArray_, pHeap;
    tSystemArray *pOldArray, *pNewArray;
    U32 newSize, oldSize;
    tMD_TypeDef *pArrayTypeDef;

    void ** p = (void**)pParams;
    ppArray_ = *(HEAP_PTR**)p++;
    newSize = *(U32*)p++;

    pOldArray = (tSystemArray*)*ppArray_;
    U32 rank = *((&(pOldArray->rank)) + 1);
    int len = *((&(pOldArray->rank)) + 2);;
    oldSize = len;

    if (oldSize == newSize) {
        // Do nothing if new length equals the current length.
        return NULL;
    }

    pArrayTypeDef = Heap_GetType(*ppArray_);
    pHeap = SystemArray_NewVector(pArrayTypeDef, rank, &newSize);
    pNewArray = (tSystemArray*)pHeap;
    *ppArray_ = pHeap;
    PTR beginning_of_elements = pNewArray->ptr_elements;
    PTR beginning_of_elements_old = pOldArray->ptr_elements;
    memcpy(beginning_of_elements, beginning_of_elements_old,
        pArrayTypeDef->pArrayElementType->arrayElementSize * ((newSize<oldSize)?newSize:oldSize));

    return NULL;
}

function_space_specifier tAsyncCall* System_Array_Reverse(PTR pThis_, PTR pParams, PTR pReturnValue) {
    tSystemArray *pArray;
    U32 index, length, elementSize, i, dec;
    tMD_TypeDef *pArrayType;
    U8 *pE1, *pE2;

    pArray = INTERNALCALL_PARAM(0, tSystemArray*);
    index = INTERNALCALL_PARAM(4, U32);
    length = INTERNALCALL_PARAM(8, U32);

    pArrayType = Heap_GetType((HEAP_PTR)pArray);
    elementSize = pArrayType->pArrayElementType->arrayElementSize;
    
    PTR beginning_of_elements = pArray->ptr_elements;
    pE1 = beginning_of_elements + index * elementSize;
    pE2 = beginning_of_elements + (index + length - 1) * elementSize;
    dec = elementSize << 1;

    while (pE2 > pE1) {
        for (i=elementSize; i>0; i--) {
            U8 c = *pE1;
            *pE1++ = *pE2;
            *pE2++ = c;
        }
        pE2 -= dec;
    }

    return NULL;
}

function_space_specifier HEAP_PTR SystemArray_NewVector(tMD_TypeDef *pArrayTypeDef, U32 rank, U32* lengths) {
    U32 heapSize;
    tSystemArray *pArray;
    // The size of an array depends on the rank.
    heapSize = sizeof(void*); // ptr to first element.
    int next = sizeof(I64); // for rank
    heapSize += next;
    next = sizeof(I64) * rank;
    heapSize += next;
    next = 1;
    for (int i = 0; i < rank; ++i) next *= lengths[i];
    next = next * pArrayTypeDef->pArrayElementType->arrayElementSize;
    heapSize += next;
    pArray = (tSystemArray*)Heap_Alloc(pArrayTypeDef, heapSize);
    pArray->ptr_elements = (PTR)((&(pArray->rank)) + 1 + rank);
    pArray->rank = rank;
    for (int i = 0; i < rank; ++i)
    {
        *((&(pArray->rank)) + 1 + i) = *lengths;
    }
    return (HEAP_PTR)pArray;
}

function_space_specifier void SystemArray_StoreElement(HEAP_PTR pThis_, U32 index, PTR value) {
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayTypeDef;
    U32 elemSize;

#if defined(_MSC_VER) && defined(_DEBUG)
    // Do a bounds check
    U32 len = *((&(pArray->rank)) + 1);
    if (index >= len) {
//      printf("SystemArray_StoreElement() Bounds check failed. Array length: %d  index: %d\n", pArray->length, index);
        __debugbreak();
    }
#endif

    pArrayTypeDef = Heap_GetType(pThis_);
    elemSize = pArrayTypeDef->pArrayElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    switch (elemSize) {
    case 1:
        ((U8*)(beginning_of_elements))[index] = *(U8*)value;
        break;
    case 2:
        ((U16*)(beginning_of_elements))[index] = *(U16*)value;
        break;
    case 4:
        ((U32*)(beginning_of_elements))[index] = *(U32*)value;
        break;
    default:
        memcpy(&beginning_of_elements[index * elemSize], value, elemSize);
        break;
    }
}

function_space_specifier void SystemArray_LoadElement(HEAP_PTR pThis_, U32 index, PTR value) {
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayTypeDef;
    U32 elemSize;

    pArrayTypeDef = Heap_GetType(pThis_);
    elemSize = pArrayTypeDef->pArrayElementType->arrayElementSize;
    PTR beginning_of_elements = pArray->ptr_elements;
    switch (elemSize) {
    case 1:
        *(U8*)value =((U8*)(beginning_of_elements))[index];
        break;
    case 2:
        *(U16*)value = ((U16*)(beginning_of_elements))[index];
        break;
    case 4:
        *(U32*)value = ((U32*)(beginning_of_elements))[index];
        break;
    default:
        memcpy(value, &beginning_of_elements[index * elemSize], elemSize);
        break;
    }
}

function_space_specifier void SystemArray_LoadElementIndices(HEAP_PTR pThis_, U64* indices, U64* value)
{
#ifdef  __CUDA_ARCH__
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
#else
    int threadId = 0;
#endif
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayTypeDef;

    pArrayTypeDef = Heap_GetType(pThis_);
    U32 elemSize = pArrayTypeDef->pArrayElementType->arrayElementSize;
	int rank = pArray->rank;
    PTR beginning_of_elements = pArray->ptr_elements;
    PTR b1 = (PTR)&pArray->rank;
    U64 * beginning_of_lengths = ((U64*)b1) + 1;

	U64 index = 0;
	for (int d = 0; d < rank; ++d)
	{
		U64 x = beginning_of_lengths[d];
		U32 y = (U32)x;
		index = index * y;
		index = index + indices[d];
	}

    switch (elemSize)
    {
        case 1:
        {
            *(U8*)value = *(((U8*)(beginning_of_elements)) + index);
            break;
        }
        case 2:
        {
            *(U16*)value = *(((U16*)(beginning_of_elements)) + index);
            break;
        }
        case 4:
        {
            *(U32*)value = *(((U32*)(beginning_of_elements)) + index);
            break;
        }
        case 8:
        {
            *(U64*)value = *(((U64*)(beginning_of_elements)) + index);
            break;
        }
        default:
        {
            memcpy(value, &beginning_of_elements[index * elemSize], elemSize);
            break;
        }
    }
}

function_space_specifier void SystemArray_StoreElementIndices(HEAP_PTR pThis_, U64* indices, U64* value)
{
#ifdef  __CUDA_ARCH__
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
#else
    int threadId = 0;
#endif
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayTypeDef;

    pArrayTypeDef = Heap_GetType(pThis_);
    U32 elemSize = pArrayTypeDef->pArrayElementType->arrayElementSize;

    PTR beginning_of_elements = pArray->ptr_elements;
	int rank = pArray->rank;
    PTR b1 = (PTR)&pArray->rank;
    U64 * beginning_of_lengths = ((U64*)b1) + 1;

	U64 index = 0;
	for (int d = 0; d < rank; ++d)
	{
		U64 x = beginning_of_lengths[d];
		U32 y = (U32)x;
		index = index * y;
		index = index + indices[d];
	}

    switch (elemSize)
    {
        case 1:
        {
            *(((U8*)(beginning_of_elements)) + index) = *(U8*)value;
            break;
        }
        case 2:
        {
            *(((U16*)(beginning_of_elements)) + index) = *(U16*)value;
            break;
        }
        case 4:
        {
            U32 v = *(U32*)value;
            *(((U32*)(beginning_of_elements)) + index) = v;
            break;
        }
        case 8:
        {
            *(((U64*)(beginning_of_elements)) + index) = *(U64*)value;
            break;
        }
        default:
        {
            memcpy(&beginning_of_elements[index * elemSize], value, elemSize);
            break;
        }
    }
}

function_space_specifier void SystemArray_LoadElementIndicesAddress(HEAP_PTR pThis_, U64* indices, HEAP_PTR * value_address)
{
#ifdef  __CUDA_ARCH__
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
#else
    int threadId = 0;
#endif
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayTypeDef;

    pArrayTypeDef = Heap_GetType(pThis_);
    U32 element_size = pArrayTypeDef->pArrayElementType->arrayElementSize;
	int rank = pArray->rank;
    PTR beginning_of_elements = pArray->ptr_elements;
    PTR b1 = (PTR)&pArray->rank;
    U64 * beginning_of_lengths = ((U64*)b1) + 1;

    U64 index = 0;
	for (int d = 0; d < rank; ++d)
	{
		U64 x = beginning_of_lengths[d];
		U32 y = (U32)x;
		index = index * y;
		index = index + indices[d];
    }
    *value_address = (((U8*)(beginning_of_elements)) + index * element_size);
}

function_space_specifier PTR SystemArray_LoadElementAddress(HEAP_PTR pThis_, U32 index) {
    tSystemArray *pArray = (tSystemArray*)pThis_;
    tMD_TypeDef *pArrayTypeDef;

#if defined(_MSC_VER) && defined(_DEBUG)
    U32 len = *((&(pArray->rank)) + 1);
    if (index >= len) {
//      printf("SystemArray_LoadElementAddress() Bounds check failed\n");
        __debugbreak();
    }
#endif

    pArrayTypeDef = Heap_GetType(pThis_);
    PTR beginning_of_elements = pArray->ptr_elements;
    return beginning_of_elements + pArrayTypeDef->pArrayElementType->arrayElementSize * index;
}

function_space_specifier U32 SystemArray_GetNumBytes(HEAP_PTR pThis_, tMD_TypeDef *pElementType) {
    U32 len = *((&(((tSystemArray*)pThis_)->rank)) + 1);
    return (len * pElementType->arrayElementSize) + sizeof(tSystemArray);
}

function_space_specifier int SystemArray_GetRank(HEAP_PTR pThis_)
{
	tSystemArray *pArray = (tSystemArray*)pThis_;
	U64 p_len = pArray->rank;
	return p_len;
}

function_space_specifier void SystemArray_SetRank(HEAP_PTR pThis_, int rank)
{
	tSystemArray *pArray = (tSystemArray*)pThis_;
	pArray->rank = rank;
}

function_space_specifier U64* SystemArray_GetDims(HEAP_PTR pThis_)
{
    tSystemArray *pArray = (tSystemArray*)pThis_;
    PTR beginning_of_elements = pArray->ptr_elements;
    PTR b1 = (PTR)&pArray->rank;
    U64 * beginning_of_lengths = ((U64*)b1) + 1;
    return beginning_of_lengths;
}