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

#include "System.String.h"

#include "MetaData.h"
#include "Types.h"
#include "Heap.h"
#include "Type.h"
#include "System.Array.h"
#include "Gstring.h"

typedef struct tSystemString_ tSystemString;
// This structure must tie up with string.cs
struct tSystemString_
{
	// Length in characters (not bytes)
	U32 length;
	// The characters
	U16 chars[0];
};

//extern function_space_specifier int CorLibDone;

// length in characters, not bytes
static function_space_specifier tSystemString* CreateStringHeapObj(U32 len)
{
	tSystemString *pSystemString;
	U32 totalSize;
	
	totalSize = sizeof(tSystemString) + (len << 1);
	pSystemString = (tSystemString*)Heap_Alloc(_bcl_->types[TYPE_SYSTEM_STRING], totalSize);
	pSystemString->length = len;
	return pSystemString;
}

function_space_specifier tAsyncCall* System_String_ctor_CharInt32(PTR pThis_, PTR pParams, PTR pReturnValue)
{
	tSystemString *pSystemString;
	U32 i;

	void **p = (void**)pParams;
	CHAR2 c = *(CHAR2*)p++;
	U32 len = *(U32*)p++;
	pSystemString = CreateStringHeapObj(len);
	for (i=0; i<len; i++) {
		pSystemString->chars[i] = c;
	}
	*(HEAP_PTR*)pReturnValue = (HEAP_PTR)pSystemString;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_ctor_CharAIntInt(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pSystemString;
	PTR charElements;

	void** p = (void**)pParams;
	HEAP_PTR charArray = *(HEAP_PTR*)p++;
	U32 startIndex = *(U32*)p++;
	U32 length = *(U32*)p++;

	charElements = SystemArray_GetElements(charArray);
	pSystemString = CreateStringHeapObj(length);
	memcpy(pSystemString->chars, charElements + (startIndex << 1), length << 1);
	*(HEAP_PTR*)pReturnValue = (HEAP_PTR)pSystemString;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_ctor_StringIntInt(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis, *pStr;
	U32 startIndex, length;

	void **p = (void**)pParams;
	pStr = *(tSystemString**)p;
	void * ppp = *p;

	p++;
	startIndex = *(U32*)p++;
	length = *(U32*)p++;

	pThis = CreateStringHeapObj(length);
	memcpy(pThis->chars, &pStr->chars[startIndex], length << 1);

	unsigned char * pc = (unsigned char *)&pThis->chars;
	for (int i = 0; i < length << 1; ++i)
	{
		unsigned char c = *pc++;
		Gprintf("c[%d] = %x\n", i, c);
	}
	*(HEAP_PTR*)pReturnValue = (HEAP_PTR)pThis;
	return NULL;
}

function_space_specifier tAsyncCall* System_String_get_Chars(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis = (tSystemString*)pThis_;
	U32 index;

	void **p = (void**)pParams;
	index = *(U32*)p++;
	*(U32*)pReturnValue = pThis->chars[index];

	return NULL;
}

function_space_specifier tAsyncCall* System_String_InternalConcat(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *s0, *s1, *ret;

	void **p = (void**)pParams;
	s0 = *(tSystemString**)p++;
	s1 = *(tSystemString**)p++;
	ret = CreateStringHeapObj(s0->length + s1->length);
	memcpy(ret->chars, s0->chars, s0->length << 1);
	memcpy(&ret->chars[s0->length], s1->chars, s1->length << 1);
	*(HEAP_PTR*)pReturnValue = (HEAP_PTR)ret;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_InternalTrim(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis = (tSystemString*)pThis_;
	HEAP_PTR pWhiteChars;
	U32 trimType, i, j, checkCharsLen;
	U32 ofsStart, ofsEnd;
	U16 *pCheckChars;
	U32 isWhiteSpace;
	tSystemString *pRet;
	U16 c;

	void **p = (void**)pParams;
	pWhiteChars = *(HEAP_PTR*)p++;
	trimType = *(U32*)p++;
	pCheckChars = (U16*)SystemArray_GetElements(pWhiteChars);
	checkCharsLen = SystemArray_GetLength(pWhiteChars);

	ofsStart = 0;
	ofsEnd = pThis->length;
	if (trimType & 1) {
		// Trim start
		for (i=ofsStart; i<ofsEnd; i++) {
			// Check if each char is in the array
			isWhiteSpace = 0;
			c = pThis->chars[i];
			for (j=0; j<checkCharsLen; j++) {
				if (c == pCheckChars[j]) {
					isWhiteSpace = 1;
					break;
				}
			}
			if (isWhiteSpace == 0) {
				ofsStart = i;
				break;
			}
		}
	}
	if (trimType & 2) {
		// Trim end
		for (i=ofsEnd-1; i>=ofsStart; i--) {
			// Check if each char is in the array
			isWhiteSpace = 0;
			c = pThis->chars[i];
			for (j=0; j<checkCharsLen; j++) {
				if (c == pCheckChars[j]) {
					isWhiteSpace = 1;
					break;
				}
			}
			if (isWhiteSpace == 0) {
				ofsEnd = i + 1;
				break;
			}
		}
	}

	pRet = CreateStringHeapObj(ofsEnd - ofsStart);
	memcpy(pRet->chars, &pThis->chars[ofsStart], (ofsEnd - ofsStart) << 1);
	*(HEAP_PTR*)pReturnValue = (HEAP_PTR)pRet;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_Equals(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *a, *b;
	U32 ret;

	void **p = (void**)pParams;
	a = *(tSystemString**)p++;
	b = *(tSystemString**)p++;

	if (a == b) {
		ret = 1;
	} else if (a == NULL || b == NULL || a->length != b->length) {
		ret = 0;
	} else {
		ret = (Gmemcmp(a->chars, b->chars, a->length<<1) == 0)?1:0;
	}
	*(U32*)pReturnValue = ret;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_GetHashCode(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis = (tSystemString*)pThis_;
	U16 *pChar, *pEnd;
	I32 hash;
	
	hash = 0;
	pChar = pThis->chars;
	pEnd = pChar + pThis->length - 1;
	for (; pChar < pEnd; pChar += 2) {
		hash = (hash << 5) - hash + pChar[0];
		hash = (hash << 5) - hash + pChar[1];
	}
	if (pChar <= pEnd) {
		hash = (hash << 5) - hash + pChar[0];
	}
	*(I32*)pReturnValue = hash;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_InternalReplace(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis = (tSystemString*)pThis_;

	void **p = (void**)pParams;
	tSystemString *pOld = *(tSystemString**)p++;
	tSystemString *pNew = *(tSystemString**)p++;
	tSystemString *pResult;
	U32 thisLen, oldLen, newLen;
	U16 *pThisChar0, *pOldChar0, *pNewChar0, *pResultChar0;
	U32 i, j, replacements, dstIndex;
	U32 resultLen;

	// This function (correctly) assumes that the old string is not empty
	thisLen = pThis->length;
	oldLen = pOld->length;
	newLen = pNew->length;
	pThisChar0 = pThis->chars;
	pOldChar0 = pOld->chars;
	pNewChar0 = pNew->chars;

	replacements = 0;
	for (i=0; i<thisLen-oldLen+1; i++) {
		U32 match = 1;
		for (j=0; j<oldLen; j++) {
			if (pThisChar0[i+j] != pOldChar0[j]) {
				match = 0;
				break;
			}
		}
		if (match) {
			i += oldLen - 1;
			replacements++;
		}
	}
	resultLen = thisLen - (oldLen - newLen) * replacements;
	pResult = CreateStringHeapObj(resultLen);
	pResultChar0 = pResult->chars;
	dstIndex = 0;
	for (i=0; i<thisLen; i++) {
		U32 match;
		if (i<thisLen-oldLen+1) {
			for (j=0; j<oldLen; j++) {
				match = 1;
				if (pThisChar0[i+j] != pOldChar0[j]) {
					match = 0;
					break;
				}
			}
		} else {
			match = 0;
		}
		if (match) {
			memcpy(&pResultChar0[dstIndex], pNewChar0, newLen << 1);
			dstIndex += newLen;
			i += oldLen - 1;
		} else {
			pResultChar0[dstIndex++] = pThisChar0[i];
		}
	}
	*(HEAP_PTR*)pReturnValue = (HEAP_PTR)pResult;

	return NULL;
}

function_space_specifier tAsyncCall* System_String_InternalIndexOf(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis = (tSystemString*)pThis_;
	void ** p = (void**)pParams;
	U16 value = *(U16*)p++;
	I32 startIndex = *(I32*)p++;
	I32 count = *(I32*)p++;
	U32 forwards = *(U32*)p;

	I32 lastIndex;
	I32 inc;
	I32 i;

	if (forwards) {
		lastIndex = startIndex + count;
		inc = 1;
		i = startIndex;
	} else {
		lastIndex = startIndex - 1;
		inc = -1;
		i = startIndex + count - 1;
	}
	for (; i != lastIndex; i += inc) {
		if (pThis->chars[i] == value) {
			*(I32*)pReturnValue = i;
			return NULL;
		}
	}
	*(I32*)pReturnValue = -1;
	return NULL;
}

function_space_specifier tAsyncCall* System_String_InternalIndexOfAny(PTR pThis_, PTR pParams, PTR pReturnValue) {
	tSystemString *pThis = (tSystemString*)pThis_;

	void **p = (void**)pParams;
	HEAP_PTR valueArray = *(HEAP_PTR*)p++;
	I32 startIndex = *(I32*)p++;
	I32 count = *(I32*)p++;
	U32 forwards = *(U32*)p++;

	PTR valueChars = SystemArray_GetElements(valueArray);
	U32 numValueChars = SystemArray_GetLength(valueArray);

	I32 lastIndex;
	I32 inc;
	I32 i, j;

	if (forwards) {
		lastIndex = startIndex + count;
		inc = 1;
		i = startIndex;
	} else {
		lastIndex = startIndex - 1;
		inc = -1;
		i = startIndex + count - 1;
	}
	for (; i != lastIndex; i += inc) {
		U16 thisChar = pThis->chars[i];
		for (j=numValueChars - 1; j>=0; j--) {
			if (thisChar == ((U16*)valueChars)[j]) {
				*(I32*)pReturnValue = i;
				return NULL;
			}
		}
	}
	*(I32*)pReturnValue = -1;
	return NULL;
}

function_space_specifier HEAP_PTR SystemString_FromUserStrings(tMetaData *pMetaData, IDX_USERSTRINGS index) {
	unsigned int stringLen;
	STRING2 string;
	tSystemString *pSystemString;
	
	string = MetaData_GetUserString(pMetaData, index, &stringLen);
	// Note: stringLen is in bytes
	pSystemString = (tSystemString*)CreateStringHeapObj(stringLen >> 1);
	memcpy(pSystemString->chars, string, stringLen);
	return (HEAP_PTR)pSystemString;
}

function_space_specifier HEAP_PTR SystemString_FromCharPtrASCII(char *pStr) {
	int stringLen, i;
	tSystemString *pSystemString;

	stringLen = (int)Gstrlen(pStr);
	pSystemString = CreateStringHeapObj(stringLen);
	for (i=0; i<stringLen; i++) {
		pSystemString->chars[i] = pStr[i];
	}
	return (HEAP_PTR)pSystemString;
}

function_space_specifier HEAP_PTR SystemString_FromCharPtrUTF16(U16 *pStr) {
	tSystemString *pSystemString;
	int strLen = 0;

	while (pStr[strLen] != 0) {
		strLen++;
	}
	pSystemString = CreateStringHeapObj(strLen);
	memcpy(pSystemString->chars, pStr, strLen << 1);
	return (HEAP_PTR)pSystemString;
}

function_space_specifier HEAP_PTR Internal_SystemString_FromCharPtrUTF16(int strLen, U16 *pStr)
{
	tSystemString *pSystemString;
	pSystemString = CreateStringHeapObj(strLen);
	memcpy(pSystemString->chars, pStr, strLen << 1);
	return (HEAP_PTR)pSystemString;
}

function_space_specifier STRING2 SystemString_GetString(HEAP_PTR pThis_, U32 *pLength) {
	tSystemString *pThis = (tSystemString*)pThis_;

	if (pLength != NULL) {
		*pLength = pThis->length;
	}
	return pThis->chars;
}

function_space_specifier U32 SystemString_GetNumBytes(HEAP_PTR pThis_) {
	return (((tSystemString*)pThis_)->length << 1) + sizeof(tSystemString);
}