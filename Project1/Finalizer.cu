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

#include "Finalizer.h"

__device__ static HEAP_PTR *ppToFinalize;
__device__ static int toFinalizeOfs, toFinalizeCapacity;

__device__ void Finalizer_Init() {
	toFinalizeCapacity = 4;
	ppToFinalize = (HEAP_PTR*)malloc(toFinalizeCapacity * sizeof(void*));
	toFinalizeOfs = 0;
}

__device__ void AddFinalizer(HEAP_PTR ptr) {
	if (toFinalizeOfs >= toFinalizeCapacity) {
		toFinalizeCapacity <<= 1;
		ppToFinalize = (HEAP_PTR*)gpurealloc(ppToFinalize, toFinalizeCapacity * sizeof(void*));
	}
	ppToFinalize[toFinalizeOfs++] = ptr;
}

__device__ HEAP_PTR GetNextFinalizer() {
	if (toFinalizeOfs == 0) {
		return NULL;
	}
	return ppToFinalize[--toFinalizeOfs];
}