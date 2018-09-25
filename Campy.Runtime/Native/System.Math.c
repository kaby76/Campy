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
#include "Type.h"

#include "System.Math.h"

#include <math.h>
#include "Gprintf.h"

function_space_specifier tAsyncCall* System_Math_Acos(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = acos(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Acosh(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = acosh(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Asin(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = asin(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Asinh(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = asinh(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Atan(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = asin(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Atan2(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = asin(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Atanh(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = atanh(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Cbrt(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = cbrt(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Ceiling(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = ceil(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Cos(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = cos(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Cosh(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = cosh(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Exp(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = exp(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Floor(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = floor(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Log(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = log(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Log10(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = log10(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Pow(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d1 = *(double*)p;
	++p;
	double d2 = *(double*)p;
	*(double*)pReturnValue = pow(d1, d2);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Sin(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = sin(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Sinh(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = sinh(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Sqrt(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = sqrt(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Tan(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = tan(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_Tanh(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d = *(double*)p;
	*(double*)pReturnValue = tanh(d);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_FMod(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d1 = *(double*)p;
	++p;
	double d2 = *(double*)p;
	*(double*)pReturnValue = fmod(d1, d2);
    return NULL;
}

function_space_specifier tAsyncCall* System_Math_ModF(PTR pThis_, PTR pParams, PTR pReturnValue) {
	void** p = (void**)pParams;
	double d1 = *(double*)p;
	++p;
	double * d2 = (double*)p;
	*(double*)pReturnValue = modf(d1, d2);
    return NULL;
}

