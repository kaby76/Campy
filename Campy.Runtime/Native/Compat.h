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

#if !defined (__COMPAT_H)
#define __COMPAT_H


// Define CUDA spaces. Sometimes we don't want to compile under Cuda.
// But, if you do, #define CUDA
#if defined(CUDA)
#include <cuda.h>
#define gpu_space_specifier __device__
#define host_space_specifier __host__
#define global_space_specifier __global__
#define function_space_specifier gpu_space_specifier host_space_specifier
#else
#define gpu_space_specifier __device__
#define host_space_specifier __host__
#define global_space_specifier __global__
#define function_space_specifier gpu_space_specifier host_space_specifier
#define __device__
#define __global__
#define __host__
#endif

#include <stdio.h>
#include <stdarg.h>
//#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef XXXXXXX
#include <strings.h>
#include <sys/uio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <dlfcn.h>
#include <glob.h>
#endif

#ifdef WIN32
#include <winsock2.h> // winsock2.h must be included before windows.h
#include <io.h>
#include <windows.h>
#include <conio.h>
// Disable warning about deprecated functions
#pragma warning(disable:4996)
// Disable warning about converting pointer to int
#pragma warning(disable:4311)
// Disable warning about converting int to pointer
#pragma warning(disable:4312)
// convert warning to error about not all control paths return value in a function
#pragma warning(error:4715)
// convert warning to error about no return value in a function
#pragma warning(error:4716)
// convert warning to error about function must return a value
#pragma warning(error:4033)
// convert warning to error about no pointer mismatch
#pragma warning(error:4022)
// convert warning to error about pointer differs in indirection
#pragma warning(error:4047)
// convert warning to error about function undefined
#pragma warning(error:4013)
// convert warning to error about too many parameters to function call
#pragma warning(error:4020)
// convert warning to error about incompatible types
#pragma warning(error:4133)
// convert warning to error about different types for parameters in function call
#pragma warning(error:4024)
// convert warning to error about different parameter lists
#pragma warning(error:4113)
// convert warning to error about macro not enough parameters
#pragma warning(error:4003)

#define strcasecmp stricmp

#define LIB_PREFIX ""
#define LIB_SUFFIX "dll"

#else // WIN32

#include <stdlib.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <glob.h>
#include <stdint.h>

#define O_BINARY 0
#define LIB_PREFIX "./"
#define LIB_SUFFIX "so"

#endif // WIN32


#define TMALLOC(t) (t*)Gmalloc(sizeof(t))
#define TMALLOCFOREVER(t) (t*)mallocForever(sizeof(t))


function_space_specifier  void gpuexit(int _Code);

function_space_specifier size_t Gstrlen(
	char const* _Str
	);

function_space_specifier int Gsprintf(
	char*       const _Buffer,
	char const* const _Format,
	...);

function_space_specifier void* Grealloc(
	void*  _Block,
	size_t _Size
	);

function_space_specifier void* Gmalloc(
	size_t _Size
	);

function_space_specifier void Gfree(
	void*  _Block
	);


// Macro to help dereference parameters that are passed into an internal system
// call. pParams is assumed to be an array of void*, with each element type casted
// to/from the type desired. The old pParams[i] mechanism does not work.
#define INTERNALSYSTEMCALL_PARAM(pParams, ofs, type) *(type*)(((void**)pParams) + ofs)

#endif

