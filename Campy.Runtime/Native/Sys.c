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

#include "MetaData.h"
#include "Types.h"
#include "Gstring.h"
#include "Gprintf.h"
#include <stdio.h>

#ifdef WIN32
namespace dbg {
	void fail(const char* func, const char* msg);
}
#endif

function_space_specifier void Crash(const char *pMsg, ...) {
	va_list va;

	Gprintf("\n\n*** CRASH ***\n");

#ifdef WIN32
	dbg::fail("crash", "crash");
#endif
	va_start(va, pMsg);
	char buf[10000];
	Gvsprintf(buf, pMsg, va);
	va_end(va);
	Gprintf(buf);
	Gprintf("\n\n");

#ifdef WIN32
	{
		// Cause a delibrate exception, to get into debugger
		__debugbreak();
	}
#endif
//	gpuexit(1);
}

//function_space_specifier U32 logLevel = 0;

function_space_specifier void log_f(U32 level, const char *pMsg, ...) {
	va_list va;
	if (_bcl_->logLevel >= level) {
		va_start(va, pMsg);
		//Gvprintf(pMsg, va);
		va_end(va);
	}
}

//function_space_specifier static char methodName[2048];
function_space_specifier char* Sys_GetMethodDesc(tMD_MethodDef *pMethod) {
	U32 i;

	Gsprintf(_bcl_->methodName, "%s.%s.%s(", pMethod->pParentType->nameSpace, pMethod->pParentType->name, pMethod->name);
	for (i=METHOD_ISSTATIC(pMethod)?0:1; i<pMethod->numberOfParameters; i++) {
		if (i > (U32)(METHOD_ISSTATIC(pMethod)?0:1)) {
			Gsprintf(Gstrchr(_bcl_->methodName, 0), ",");
		}
		Gsprintf(Gstrchr(_bcl_->methodName, 0), pMethod->pParams[i].pTypeDef->name);
	}
	Gsprintf(Gstrchr(_bcl_->methodName, 0), ")");
	return _bcl_->methodName;
}

// function_space_specifier static U32 mallocForeverSize = 0;
// malloc() some memory that will never need to be resized or freed.
function_space_specifier void* mallocForever(U32 size) {
	_bcl_->mallocForeverSize += size;
log_f(3, "--- mallocForever: TotalSize %d\n", _bcl_->mallocForeverSize);
	return Gmalloc(size);
}

/*
#ifdef _DEBUG
void* mallocTrace(int s, char *pFile, int line) {
	//printf("MALLOC: %s:%d %d\n", pFile, line, s);
#undef malloc
	return malloc(s);
}
#endif
*/

function_space_specifier U64 msTime() {
//#ifdef WIN32
//	static LARGE_INTEGER freq = {0,0};
//	LARGE_INTEGER time;
//	if (freq.QuadPart == 0) {
//		QueryPerformanceFrequency(&freq);
//	}
//	QueryPerformanceCounter(&time);
//	return (time.QuadPart * 1000) / freq.QuadPart;
//#else
//	struct timeval tp;
//	U64 ms;
//	gettimeofday(&tp,NULL);
//	ms = tp.tv_sec;
//	ms *= 1000;
//	ms += ((U64)tp.tv_usec)/((U64)1000);
//	return ms;
//#endif
	return 0;
}

#if defined(DIAG_METHOD_CALLS) || defined(DIAG_OPCODE_TIMES) || defined(DIAG_GC) || defined(DIAG_TOTAL_TIME)
function_space_specifier U64 microTime() {
//#ifdef WIN32
//	static LARGE_INTEGER freq = {0,0};
//	LARGE_INTEGER time;
//	if (freq.QuadPart == 0) {
//		QueryPerformanceFrequency(&freq);
//	}
//	QueryPerformanceCounter(&time);
//	return (time.QuadPart * 1000000) / freq.QuadPart;
//#else
//	struct timeval tp;
//	U64 ms;
//	gettimeofday(&tp,NULL);
//	ms = tp.tv_sec;
//	ms *= 1000000;
//	ms += ((U64)tp.tv_usec);
//	return ms;
//#endif
	return 0;
}
#endif

function_space_specifier void SleepMS(U32 ms) {
//#ifdef WIN32
//	Sleep(ms);
//#else
//	sleep(ms / 1000);
//	usleep((ms % 1000) * 1000);
//#endif
}