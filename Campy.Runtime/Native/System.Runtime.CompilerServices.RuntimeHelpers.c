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

#include "System.Runtime.CompilerServices.RuntimeHelpers.h"

#include "MetaData.h"
#include "Types.h"
#include "Heap.h"
#include "Type.h"
#include "System.Array.h"
#include "MetaDataTables.h"

function_space_specifier tAsyncCall* System_Runtime_CompilerServices_RuntimeHelpers_InitializeArray(PTR pThis_, PTR pParams, PTR pReturnValue) {
    HEAP_PTR pArray;
    PTR pRawData;
    tMD_TypeDef *pArrayTypeDef;
    tMD_TypeDef *pDataTypeDef;
    PTR pElements;
    U32 arrayLength;

    U64 * p = (U64*)pParams;
    U64 p0 = *p;
    p++;
    U64 p1 = *p;
    pArray = ((HEAP_PTR*)pParams)[0];
    pRawData = ((PTR*)pParams)[1];
    U64 p2 = *(U64*)p1;

    // The data is encapsulated in the object's class.
    pDataTypeDef = Heap_GetType(pRawData);

    // The data is hanging off a field of the type.
    tMD_FieldDef * fd = (tMD_FieldDef*)p2;
    
    PTR mem = fd->pMemory;

    pArrayTypeDef = Heap_GetType(pArray);
    arrayLength = SystemArray_GetLength(pArray);
    pElements = SystemArray_GetElements(pArray);
    memcpy(pElements, mem, pArrayTypeDef->pArrayElementType->arrayElementSize * arrayLength);

    return NULL;
}
