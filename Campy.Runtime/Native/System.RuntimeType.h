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

#pragma once

#include "Types.h"
#include "MetaData.h"

typedef struct tRuntimeType_ tRuntimeType;
struct tRuntimeType_ {
    // The pointer to the TypeDef object of this type.
    tMD_TypeDef *pTypeDef;
};

function_space_specifier tAsyncCall* System_RuntimeType_get_Name(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_get_Namespace(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_GetNestingParentType(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_get_BaseType(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_get_IsEnum(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_get_IsGenericType(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_Internal_GetGenericTypeDefinition(PTR pThis_, PTR pParams, PTR pReturnValue);
function_space_specifier tAsyncCall* System_RuntimeType_GetGenericArguments(PTR pThis_, PTR pParams, PTR pReturnValue);

// Create a new heap object which is the RuntimeType object for the given type.
function_space_specifier HEAP_PTR RuntimeType_New(tMD_TypeDef *pTypeDef);
// De-ref a runtime type pointer
function_space_specifier tMD_TypeDef* RuntimeType_DeRef(PTR type);
