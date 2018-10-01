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

#include "System.Threading.Interlocked.h"

function_space_specifier tAsyncCall* System_Threading_Interlocked_CompareExchange_Int32(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    U32 *pLoc = INTERNALCALL_PARAM(0, U32*);
    U32 value = INTERNALCALL_PARAM(1, U32);
    U32 comparand = INTERNALCALL_PARAM(2, U32);
	U32 result;
#ifdef  __CUDA_ARCH__
	result = atomicCAS((U32*)pLoc, (U32)comparand, (U32)value);
#else
	result = (U32)(*(pLoc));
	if (*pLoc == comparand)
	{
		*pLoc = value;
	}
#endif
	*(U32*)pReturnValue = result;

    return NULL;
}

function_space_specifier tAsyncCall* System_Threading_Interlocked_Increment_Int32(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    I32 *pLoc = INTERNALCALL_PARAM(0, I32*);
	I32 result;
#ifdef  __CUDA_ARCH__
	result = atomicAdd((U32*)pLoc, (U32)1);
	result += 1; // follow semantics of System.Threading.Interlocked::Increment().
#else
	(*pLoc)++;
	result = *pLoc;
#endif
    *(I32*)pReturnValue = result;
    return NULL;
}

function_space_specifier tAsyncCall* System_Threading_Interlocked_Decrement_Int32(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    I32 *pLoc = INTERNALCALL_PARAM(0, I32*);
	I32 result;
#ifdef  __CUDA_ARCH__
	result = atomicSub((U32*)pLoc, (U32)1);
	result -= 1; // follow semantics of System.Threading.Interlocked::Increment().
#else
	(*pLoc)--;
	result = *pLoc;
#endif
	*(I32*)pReturnValue = result;
    return NULL;
}

function_space_specifier tAsyncCall* System_Threading_Interlocked_Add_Int32(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    U32 *pLoc = INTERNALCALL_PARAM(0, U32*);
    U32 value = INTERNALCALL_PARAM(1, U32);
	I32 result;
#ifdef  __CUDA_ARCH__
	result = atomicAdd((U32*)pLoc, (U32)value);
	result += value; // follow semantics of System.Threading.Interlocked::Increment().
#else
	(*pLoc) += value;
	result = *pLoc;
#endif
	*(I32*)pReturnValue = result;
    return NULL;
}

function_space_specifier tAsyncCall* System_Threading_Interlocked_Exchange_Int32(PTR pThis_, PTR pParams, PTR pReturnValue)
{
    U32 *pLoc = INTERNALCALL_PARAM(0, U32*);
    U32 value = INTERNALCALL_PARAM(1, U32);
	I32 result;
#ifdef  __CUDA_ARCH__
	result = atomicExch((U32*)pLoc, (U32)value);
#else
	result = *pLoc;
	*pLoc = value;
#endif
	*(I32*)pReturnValue = result;
    return NULL;
}
