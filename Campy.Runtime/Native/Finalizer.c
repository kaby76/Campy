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

//function_space_specifier static HEAP_PTR *ppToFinalize;
//function_space_specifier static int toFinalizeOfs, toFinalizeCapacity;

function_space_specifier void Finalizer_Init() {
	_bcl_->toFinalizeCapacity = 4;
	_bcl_->ppToFinalize = (HEAP_PTR*)Gmalloc(_bcl_->toFinalizeCapacity * sizeof(void*));
	_bcl_->toFinalizeOfs = 0;
}

function_space_specifier void AddFinalizer(HEAP_PTR ptr) {
	if (_bcl_->toFinalizeOfs >= _bcl_->toFinalizeCapacity) {
		_bcl_->toFinalizeCapacity <<= 1;
		_bcl_->ppToFinalize = (HEAP_PTR*)Grealloc(_bcl_->ppToFinalize, _bcl_->toFinalizeCapacity * sizeof(void*));
	}
	_bcl_->ppToFinalize[_bcl_->toFinalizeOfs++] = ptr;
}

function_space_specifier HEAP_PTR GetNextFinalizer() {
	if (_bcl_->toFinalizeOfs == 0) {
		return NULL;
	}
	return _bcl_->ppToFinalize[--_bcl_->toFinalizeOfs];
}