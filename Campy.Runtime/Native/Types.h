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

#if defined(_MSC_VER)
//  Microsoft 
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

#ifndef STDCALL
# if defined(_MSC_VER)
#   define STDCALL __stdcall
# else
#   define STDCALL
# endif
#endif


// Indexes into the user-string heap
typedef unsigned int IDX_USERSTRINGS;

// Index into a table. most significant byte stores which table, other 3 bytes store index
typedef unsigned int IDX_TABLE;

// Flag types
typedef unsigned int FLAGS32;
typedef unsigned short FLAGS16;

// Pointers
typedef unsigned char* HEAP_PTR;
typedef unsigned char* PTR;
typedef unsigned char* SIG;
typedef char* STRING; // UTF8/ASCII string
typedef unsigned short* STRING2; // UTF16 string
typedef unsigned char* BLOB_;
typedef unsigned char* GUID_;

// Int types
typedef long long I64;
typedef unsigned long long U64;

typedef uint32_t U32;
typedef int32_t I32;
typedef int16_t I16;
typedef uint16_t U16;
typedef uint8_t U8;


#ifdef _MSC_VER

typedef int I32;
typedef unsigned int U32;
typedef short I16;
typedef unsigned short U16;
typedef char I8;
typedef unsigned char U8;

#endif // WIN32

typedef union uConvDouble_ uConvDouble;
union uConvDouble_ {
	double d;
	U64 u64;
	struct {
		U32 a;
		U32 b;
	} u32;
};

typedef union uConvFloat_ uConvFloat;
union uConvFloat_ {
	float f;
	U32 u32;
};

// other types!
typedef unsigned short CHAR2;

typedef struct tAsyncCall_ tAsyncCall;

// Native function call
typedef tAsyncCall* (*fnInternalCall)(PTR pThis_, PTR pParams, PTR pReturnValue);
// Native function call check routine for blocking IO
typedef U32 (*fnInternalCallCheck)(PTR pThis_, PTR pParams, PTR pReturnValue, tAsyncCall *pAsync);

struct tAsyncCall_ {
	// If this is a sleep call, then put the sleep time in ms here.
	// -1 means it's not a sleep call. Inifite timeouts are not allowed.
	I32 sleepTime;
	// If this is a blocking IO call, then this is the function to poll to see if the result is available.
	fnInternalCallCheck checkFn;
	// A state pointer for general use for blocking IO calls.
	PTR state;
	// Not for most functions to use. Record the start time of this async call
	U64 startTime;
};
