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
#include "Type.h"
#include "RVA.h"
#include "Gstring.h"
#include <stdio.h>
#if defined(CUDA)
#include <crt/host_defines.h>
#endif

function_space_specifier unsigned int MetaData_DecodeSigEntry(SIG *pSig) {
	unsigned char a,b,c,d;
	a = *((unsigned char*)*pSig)++;
	if ((a & 0x80) == 0) {
		// 1-byte entry
		return a;
	}
	// Special case
	if (a == 0xff) {
		return 0;
	}

	b = *((unsigned char*)*pSig)++;
	if ((a & 0xc0) == 0x80) {
		// 2-byte entry
		return ((int)(a & 0x3f)) << 8 | b;
	}
	// 4-byte entry
	c = *((unsigned char*)*pSig)++;
	d = *((unsigned char*)*pSig)++;
	return ((int)(a & 0x1f)) << 24 | ((int)b) << 16 | ((int)c) << 8 | d;
}

function_space_specifier IDX_TABLE MetaData_DecodeSigEntryToken(SIG *pSig) {
	static U8 tableID[4] = {MD_TABLE_TYPEDEF, MD_TABLE_TYPEREF, MD_TABLE_TYPESPEC, 0};

	U32 entry = MetaData_DecodeSigEntry(pSig);
	return MAKE_TABLE_INDEX(tableID[entry & 0x3], entry >> 2);
}

function_space_specifier tMetaData* MetaData() {
	tMetaData *pRet = TMALLOC(tMetaData);
	memset(pRet, 0, sizeof(tMetaData));
	return pRet;
}

function_space_specifier void MetaData_LoadStrings(tMetaData *pThis, void *pStream, unsigned int streamLen) {
	pThis->strings.pStart = (unsigned char*)pStream;

	log_f(1, "Loaded strings\n");
}

function_space_specifier unsigned int MetaData_DecodeHeapEntryLength(unsigned char **ppHeapEntry) {
	return MetaData_DecodeSigEntry((SIG*)ppHeapEntry);
}

function_space_specifier void MetaData_LoadBlobs(tMetaData *pThis, void *pStream, unsigned int streamLen) {
	pThis->blobs.pStart = (unsigned char*)pStream;

	log_f(1, "Loaded blobs\n");

}

function_space_specifier void MetaData_LoadUserStrings(tMetaData *pThis, void *pStream, unsigned int streamLen) {
	pThis->userStrings.pStart = (unsigned char*)pStream;

	log_f(1, "Loaded User Strings\n");

}

function_space_specifier void MetaData_LoadGUIDs(tMetaData *pThis, void *pStream, unsigned int streamLen) {
	pThis->GUIDs.numGUIDs = streamLen / 16;

	// This is stored -16 because numbering starts from 1. This means that a simple indexing calculation
	// can be used, as if it started from 0
	pThis->GUIDs.pGUID1 = (unsigned char*)pStream;

	log_f(1, "Read %d GUIDs\n", pThis->GUIDs.numGUIDs);
}

/*
Format of definition strings:
Always 2 characters to togther. 1st character defines source, 2nd defines destination.
Sources:
	c: 8-bit value
	s: 16-bit short
	i: 32-bit int
	S: Index into string heap
	G: Index into GUID heap
	B: Index into BLOB heap
	0: Coded index: TypeDefOrRef
	1: Coded index: HasConstant
	2: Coded index: HasCustomAttribute
	3: Coded index: HasFieldMarshall
	4: Coded index: HasDeclSecurity
	5: Coded index: MemberRefParent
	6: Coded index: HasSemantics
	7: Coded index: MethodDefOrRef
	8: Coded index: MemberForwarded
	9: Coded index: Implementation
	:: Coded index: CustomAttributeType
	;: Coded index: ResolutionScope
	<: Coded index: TypeOrMethodDef
	\x00 - \x2c: Simple indexes into the respective table
	^: RVA: Convert to pointer
	x: Nothing, use 0
	m: This metadata pointer
	l: (lower case L) Boolean, is this the last entry in this table?
	I: The original table index for this table item
Destination:
	x: nowhere, ignore
	*: 32-bit index into relevant heap;
		Or coded index - MSB = which table, other 3 bytes = table index
		Or 32-bit int
		Or pointer (also RVA)
	s: 16-bit value
	c: 8-bit value
*/
function_space_specifier static const char* tableDefs[] = {
	// 0x00
	"sxS*G*GxGx",
	// 0x01
	"x*;*S*S*",
	// 0x02
	"x*m*i*S*S*0*\x04*\x06*xclcxcxcx*x*x*x*x*x*x*x*x*x*x*I*x*x*x*x*x*x*x*x*x*x*x*x*",
	// 0x03
	NULL,
	// 0x04
	"x*m*ssxsS*B*x*x*x*x*I*x*",
	// 0x05
	NULL,
	// 0x06
	"x*m*^*ssssS*B*\x08*x*x*x*x*x*x*I*x*x*x*"
#ifdef GEN_COMBINED_OPCODES
	"x*x*x*x*x*x*"
#endif
#ifdef DIAG_METHOD_CALLS
	"x*x*x*"
#endif
	,
	// 0x07
	NULL,
	// 0x08
	"ssssS*",
	// 0x09
	"\x02*0*",
	// 0x0A
	"x*5*S*B*",
	// 0x0B
	"ccccxs1*B*",
	// 0x0C
	"2*:*B*",
	// 0x0D
	NULL,
	// 0x0E
	"ssxs4*B*",
	// 0x0F
	"ssxsi*\x02*",
	// 0x10
	NULL,
	// 0x11
	"B*",
	// 0x12
	"\x02*\x14*",
	// 0x13
	NULL,
	// 0x14
	"ssxsS*0*",
	// 0x15
	"\x02*\x17*",
	// 0x16
	NULL,
	// 0x17
	"ssxsS*B*",
	// 0x18
	"ssxs\06*6*",
	// 0x19
	"\x02*7*7*",
	// 0x1A
	"S*",
	// 0x1B
	"x*m*B*",
	// 0x1C
	"ssxs8*S*\x1a*",
	// 0x1D
	"^*\x04*",
	// 0x1E
	NULL,
	// 0x1F
	NULL,
	// 0x20
	"i*ssssssssi*B*S*S*",
	// 0x21
	NULL,
	// 0x22
	NULL,
	// 0x23
	"ssssssssi*B*S*S*B*",
	// 0x24
	NULL,
	// 0x25
	NULL,
	// 0x26
	NULL,
	// 0x27
	NULL,
	// 0x28
	NULL,
	// 0x29
	"\x02*\x02*",
	// 0x2A
	"ssss<*S*",
	// 0x2B
	"x*m*7*B*",
	// 0x2C
	"\x2a*0*",
};

// Coded indexes use this lookup table.
// Note that the extra 'z' characters are important!
// (Because of how the lookup works each string must be a power of 2 in length)
function_space_specifier static const char* codedTags[] = {
	// TypeDefOrRef
	"\x02\x01\x1Bz",
	// HasConstant
	"\x04\x08\x17z",
	// HasCustomAttribute
	"\x06\x04\x01\x02\x08\x09\x0A\x00\x0E\x17\x14\x11\x1A\x1B\x20\x23\x26\x27\x28zzzzzzzzzzzzz",
	// HasFieldMarshall
	"\x04\x08",
	// HasDeclSecurity
	"\x02\x06\x20z",
	// MemberRefParent
	"z\x01\x1A\x06\x1Bzzz",
	// HasSemantics
	"\x14\x17",
	// MethodDefOrRef
	"\x06\x0A",
	// MemberForwarded
	"\x04\x06",
	// Implementation
	"\x26\x23\x27z",
	// CustomAttributeType
	"zz\x06\x0Azzzz",
	// ResolutionScope
	"\x00\x1A\x23\x01",
	// TypeOrMethodDef
	"\x02\x06",
};

function_space_specifier static unsigned char codedTagBits[] = {
	2, 2, 5, 1, 2, 3, 1, 1, 1, 2, 3, 2, 1
};

function_space_specifier static unsigned int tableRowSize[MAX_TABLES];

function_space_specifier void MetaData_Init() {
	U32 i;
	for (i=0; i<MAX_TABLES; i++) {
		tableRowSize[i] = 0;
	}
}

function_space_specifier unsigned int GetU16(unsigned char *pSource) {
	unsigned char a, b;

	a = pSource[0];
	b = pSource[1];
	return ((unsigned int)a)
	| (((unsigned int)b) << 8);
}

function_space_specifier unsigned int GetU32(unsigned char *pSource) {
	unsigned char a, b, c, d;

	a = pSource[0];
	b = pSource[1];
	c = pSource[2];
	d = pSource[3];
	return ((unsigned int)a)
	| (((unsigned int)b) << 8)
	| (((unsigned int)c) << 16)
	| (((unsigned int)d) << 24);
}

function_space_specifier unsigned long long GetU64(unsigned char *pSource) {
	unsigned char a, b, c, d, e, f, g, h;

	a = pSource[0];
	b = pSource[1];
	c = pSource[2];
	d = pSource[3];
	e = pSource[4];
	f = pSource[5];
	g = pSource[6];
	h = pSource[7];
	return ((unsigned long long)a)
		| (((unsigned long long)b) << 8)
		| (((unsigned long long)c) << 16)
		| (((unsigned long long)d) << 24)
		| (((unsigned long long)e) << 32)
		| (((unsigned long long)f) << 40)
		| (((unsigned long long)g) << 48)
		| (((unsigned long long)h) << 56);
}

function_space_specifier unsigned int CodedIndex(tMetaData *pThis, unsigned char x, unsigned char **ppSource)
{
	unsigned int v;
	unsigned char * pSource = (unsigned char *)*ppSource;
	int ofs = x - '0';
	const char* pCoding = codedTags[ofs];
	int tagBits = codedTagBits[ofs];
	unsigned char tag = *pSource & ((1 << tagBits) - 1);
	int idxIntoTableID = pCoding[tag]; // The actual table index that we're looking for
	if (idxIntoTableID < 0 || idxIntoTableID > MAX_TABLES) {
		printf("Error: Bad table index: 0x%02x\n", idxIntoTableID);
		gpuexit(1);
	}
	if (pThis->tables.codedIndex32Bit[ofs]) {
		// Use 32-bit number
		v = GetU32(pSource) >> tagBits;
		pSource += 4;
	}
	else {
		// Use 16-bit number
		v = GetU16(pSource) >> tagBits;
		pSource += 2;
	}
	v |= idxIntoTableID << 24;
	*ppSource = pSource;
	return v;
}

function_space_specifier unsigned int Coded2Index(tMetaData *pThis, int d, unsigned char **ppSource)
{
	unsigned int v;
	unsigned char * pSource = (unsigned char *)*ppSource;
	if (pThis->tables.numRows[d] < 0x10000) {
		// Use 16-bit offset
		unsigned int val = GetU16(pSource);
		v = val;
		pSource += 2;
	}
	else {
		// Use 32-bit offset
		unsigned int val = GetU32(pSource);
		v = val;
		pSource += 4;
	}
	v |= d << 24;
	*ppSource = pSource;
	return v;
}

// Reads metadata tables into structs in a platform-independent way.
function_space_specifier void ModuleTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest)
{
	// 0x00
	// original "sxS*G*GxGx",
	tMD_Module * p = (tMD_Module*)pDest;
	memset(p, 0, sizeof(tMD_Module));
	unsigned char * pSource = (unsigned char *)*ppSource;
	
	pSource += 2;

	int v;
	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;
	
	int heap_skip = (pThis->index32BitGUID) ? 4 : 2;
	v = pThis->index32BitGUID ? GetU32(pSource) : GetU16(pSource);
	pSource += heap_skip;
	p->mvID = (pThis->GUIDs.pGUID1 + ((v - 1) * 16));

	// Skip past EnCId.
	pSource += heap_skip;

	// Skip past EnCBaseId
	pSource += heap_skip;

	*ppSource = pSource;
}

function_space_specifier void TypeRefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest)
{
	// 0x01
	// original "x*;*S*S*",
	tMD_TypeRef * p = (tMD_TypeRef*)pDest;
	memset(p, 0, sizeof(tMD_TypeRef));
	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	v = CodedIndex(pThis, ';', &pSource);
	p->resolutionScope = v;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->nameSpace = (STRING)pThis->strings.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void TypeDefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// 0x02
	// original "x*m*i*S*S*0*\x04*\x06*xclcxcxcx*x*x*x*x*x*x*x*x*x*x*I*x*x*x*x*x*x*x*x*x*x*x*x*",
	// original
	// x* ptypedef
	// m* pmetadata
	// i* flags
	// S* name
	// S* namespace
	// 0* extends
	// \x04* fieldlist
	// \x06* methodlist
	// xc isfilled
	// lc islast
	// xc isvaluetype
	// xc stacktype
	// x* instancememsize
	// x* pparent
	// x* pvtable
	// x* numvirtualmethods
	// x* pstaticfields
	// x* istypeinitialize/isgenericdefintion/isprimed/padding.
	// x* pstaticconstructor
	// x* arrayelementsize
	// x* stacksize
	// x* numinterfaces
	// x* pinterfacemaps
	// I* tableindex
	// x* pgenericinstances
	// x* pgenericdefinition
	// x* ppclasstypeargs
	// x* parrayelementtype
	// x* numfields
	// x* ppfields
	// x* staticfieldsize
	// x* nummethods
	// x* ppmethods
	// x* pnestedin
	// x* pfinalizer
	// x*", typeobject
	tMD_TypeDef * p = (tMD_TypeDef*)pDest;
	memset(p, 0, sizeof(tMD_TypeDef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->flags = GetU32(pSource);
	pSource += 4;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->nameSpace = (STRING)pThis->strings.pStart + v;

	v = CodedIndex(pThis, '0', &pSource);
	p->extends = v;

	v = Coded2Index(pThis, 4, &pSource);
	p->fieldList = v;

	v = Coded2Index(pThis, 6, &pSource);
	p->methodList = v;

	p->isLast = (row == numRows - 1);
	p->pMetaData = pThis;

	p->tableIndex = MAKE_TABLE_INDEX(MD_TABLE_TYPEDEF, row + 1);

	*ppSource = pSource;
}

function_space_specifier void FieldPtrTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	tMD_TypeDef * p = (tMD_TypeDef*)pDest;
	memset(p, 0, sizeof(tMD_TypeDef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	*ppSource = pSource;
}

function_space_specifier void FieldDefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// "x*m*ssxsS*B*x*x*x*x*I*x*",

	tMD_FieldDef * p = (tMD_FieldDef*)pDest;
	memset(p, 0, sizeof(tMD_FieldDef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->flags = GetU16(pSource);
	pSource += 2;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->signature = pThis->blobs.pStart + v;

	p->pMetaData = pThis;

	p->tableIndex = MAKE_TABLE_INDEX(MD_TABLE_FIELDDEF, row + 1);

	*ppSource = pSource;
}

function_space_specifier void MethodDefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// 	"x*m*^*ssssS*B*\x08*x*x*x*x*x*x*I*x*x*x*"

	tMD_MethodDef * p = (tMD_MethodDef*)pDest;
	memset(p, 0, sizeof(tMD_MethodDef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->pMetaData = pThis;

	int v;

	v = GetU32(pSource);
	pSource += 4;
	p->pCIL = (U8*)RVA_FindData(pRVA, v);

	p->implFlags = GetU16(pSource);
	pSource += 2;

	p->flags = GetU16(pSource);
	pSource += 2;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->signature = pThis->blobs.pStart + v;

	p->paramList = Coded2Index(pThis, 8, &pSource);;

	p->tableIndex = MAKE_TABLE_INDEX(MD_TABLE_METHODDEF, row + 1);

	*ppSource = pSource;
}

function_space_specifier void ParamTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// 0x08
	// "ssssS*",

	tMD_Param * p = (tMD_Param*)pDest;
	memset(p, 0, sizeof(tMD_Param));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	v = GetU16(pSource);
	pSource += 2;
	p->flags = v;

	v = GetU16(pSource);
	pSource += 2;
	p->sequence = v;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void InterfaceImplTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x09 - InterfaceImpl
	// "\x02*0*",

	tMD_InterfaceImpl * p = (tMD_InterfaceImpl*)pDest;
	memset(p, 0, sizeof(tMD_InterfaceImpl));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->class_ = Coded2Index(pThis, 4, &pSource);

	p->interface_ = CodedIndex(pThis, '0', &pSource);

	*ppSource = pSource;
}

function_space_specifier void MemberRefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x0A - MemberRef
	// "x*5*S*B*",

	tMD_MemberRef * p = (tMD_MemberRef*)pDest;
	memset(p, 0, sizeof(tMD_MemberRef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->class_ = CodedIndex(pThis, '5', &pSource);

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->signature = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void ConstantTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x0B - Constant
	// "ccccxs1*B*",

	tMD_Constant * p = (tMD_Constant*)pDest;
	memset(p, 0, sizeof(tMD_Constant));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->type = *(U8*)pSource;
	pSource++;

	pSource++; // Skip intensional.

	p->parent = CodedIndex(pThis, '1', &pSource);

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->value = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void CustomAttributeTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x0C - CustomAttribute
	// "2*:*B*",

	tMD_CustomAttribute * p = (tMD_CustomAttribute*)pDest;
	memset(p, 0, sizeof(tMD_CustomAttribute));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->parent = CodedIndex(pThis, '2', &pSource);

	p->type = CodedIndex(pThis, ':', &pSource);

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->value = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void DeclSecurityTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x0E - DeclSecurity
	// "ssxs4*B*",

	tMD_DeclSecurity * p = (tMD_DeclSecurity*)pDest;
	memset(p, 0, sizeof(tMD_DeclSecurity));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->action = GetU16(pSource);
	pSource += 2;

	p->parent = CodedIndex(pThis, '4', &pSource);

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->permissionSet = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void ClassLayoutTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x0F - ClassLayout
	// "ssxsi*\x02*",

	tMD_ClassLayout * p = (tMD_ClassLayout*)pDest;
	memset(p, 0, sizeof(tMD_ClassLayout));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->packingSize = GetU16(pSource);
	pSource += 2;

	p->classSize = GetU32(pSource);
	pSource += 4;

	p->parent = Coded2Index(pThis, 2, &pSource);

	*ppSource = pSource;
}

function_space_specifier void StandAloneSigTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x11 - StandAloneSig
	// "B*",

	tMD_StandAloneSig * p = (tMD_StandAloneSig*)pDest;
	memset(p, 0, sizeof(tMD_StandAloneSig));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->signature = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void EventMapTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x12 - EventMap
	// "\x02*\x14*",

	tMD_EventMap * p = (tMD_EventMap*)pDest;
	memset(p, 0, sizeof(tMD_EventMap));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->parent = Coded2Index(pThis, 2, &pSource);

	p->eventList = Coded2Index(pThis, 0x14, &pSource);

	*ppSource = pSource;
}

function_space_specifier void EventTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x14 - Event
	// "ssxsS*0*",

	tMD_Event * p = (tMD_Event*)pDest;
	memset(p, 0, sizeof(tMD_Event));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int v;

	p->eventFlags = GetU16(pSource);
	pSource += 2;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	p->eventType = CodedIndex(pThis, '0', &pSource);

	*ppSource = pSource;
}

function_space_specifier void PropertyMapTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x15 - PropertyMap
	// "\x02*\x17*",

	tMD_PropertyMap * p = (tMD_PropertyMap*)pDest;
	memset(p, 0, sizeof(tMD_PropertyMap));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->parent = Coded2Index(pThis, 0x2, &pSource);

	p->propertyList = Coded2Index(pThis, 0x17, &pSource);

	*ppSource = pSource;
}

function_space_specifier void PropertyTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x17 - Property
	// "ssxsS*B*",

	tMD_Property * p = (tMD_Property*)pDest;
	memset(p, 0, sizeof(tMD_Property));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->flags = GetU16(pSource);
	pSource += 2;

	int string_skip = pThis->index32BitString ? 4 : 2;
	int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->typeSig = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void MethodSemanticsTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x18 - MethodSemantics
	// "ssxs\06*6*",

	tMD_MethodSemantics * p = (tMD_MethodSemantics*)pDest;
	memset(p, 0, sizeof(tMD_MethodSemantics));

	printf("MS %d %d\n", row, numRows);

	unsigned char * pSource = (unsigned char *)*ppSource;

	//printf("Getting 16 bit\n");
	p->semantics = GetU16(pSource);
	pSource += 2;

	printf("Getting 2 I\n");
	p->method = Coded2Index(pThis, 0x6, &pSource);

	//printf("Getting I\n");

	p->association = CodedIndex(pThis, '6', &pSource);
	printf("done.\n");
	*ppSource = pSource;
}

function_space_specifier void MethodImplTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x19 - MethodImpl
	// 	"\x02*7*7*"

	tMD_MethodImpl * p = (tMD_MethodImpl*)pDest;
	memset(p, 0, sizeof(tMD_MethodImpl));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->class_ = Coded2Index(pThis, 0x2, &pSource);

	p->methodBody = CodedIndex(pThis, '7', &pSource);

	p->methodDeclaration = CodedIndex(pThis, '7', &pSource);

	*ppSource = pSource;
}

function_space_specifier void ModuleRefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x1a - ModuleRef
	// 	"S*"

	tMD_ModuleRef * p = (tMD_ModuleRef*)pDest;
	memset(p, 0, sizeof(tMD_ModuleRef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int string_skip = pThis->index32BitString ? 4 : 2;
	unsigned int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void TypeSpecTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x1B - TypeSpec
	// 	"x*m*B*"

	tMD_TypeSpec * p = (tMD_TypeSpec*)pDest;
	memset(p, 0, sizeof(tMD_TypeSpec));

	unsigned char * pSource = (unsigned char *)*ppSource;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	unsigned int v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->signature = pThis->blobs.pStart + v;

	p->pMetaData = pThis;

	*ppSource = pSource;
}

function_space_specifier void ImplMapTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x1c - ImplMap
	// 	"ssxs8*S*\x1a*"

	tMD_ImplMap * p = (tMD_ImplMap*)pDest;
	memset(p, 0, sizeof(tMD_ImplMap));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->mappingFlags = GetU16(pSource);
	pSource += 2;

	p->memberForwarded = CodedIndex(pThis, '8', &pSource);

	int string_skip = pThis->index32BitString ? 4 : 2;
	unsigned int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->importName = (STRING)pThis->strings.pStart + v;

	p->importScope = Coded2Index(pThis, 0x1a, &pSource);

	*ppSource = pSource;
}

function_space_specifier void FieldRVATableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x1D - FieldRVA
	// "^*\x04*"

	tMD_FieldRVA * p = (tMD_FieldRVA*)pDest;
	memset(p, 0, sizeof(tMD_FieldRVA));

	unsigned char * pSource = (unsigned char *)*ppSource;

	unsigned int v = GetU32(pSource);
	pSource += 4;
	p->rva = RVA_FindData(pRVA, v);

	p->field = Coded2Index(pThis, 0x04, &pSource);

	*ppSource = pSource;
}

function_space_specifier void AssemblyTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x20 - Assembly
	// "i4s2s2s2s2i4B8S8S8"

	tMD_Assembly * p = (tMD_Assembly*)pDest;
	memset(p, 0, sizeof(tMD_Assembly));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->hashAlgID = GetU32(pSource);
	pSource += 4;

	p->majorVersion = GetU16(pSource);
	pSource += 2;

	p->minorVersion = GetU16(pSource);
	pSource += 2;

	p->buildNumber = GetU16(pSource);
	pSource += 2;

	p->revisionNumber = GetU16(pSource);
	pSource += 2;

	p->flags = GetU32(pSource);
	pSource += 4;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	unsigned int v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->publicKey = pThis->blobs.pStart + v;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->culture = (STRING)pThis->strings.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void AssemblyRefTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x23 - AssemblyRef
	// "s2s2s2s2i4B8S8S8B8"

	tMD_AssemblyRef * p = (tMD_AssemblyRef*)pDest;
	memset(p, 0, sizeof(tMD_AssemblyRef));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->majorVersion = GetU16(pSource);
	pSource += 2;

	p->minorVersion = GetU16(pSource);
	pSource += 2;

	p->buildNumber = GetU16(pSource);
	pSource += 2;

	p->revisionNumber = GetU16(pSource);
	pSource += 2;

	p->flags = GetU32(pSource);
	pSource += 4;

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	unsigned int v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->publicKeyOrToken = pThis->blobs.pStart + v;

	int string_skip = pThis->index32BitString ? 4 : 2;
	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->culture = (STRING)pThis->strings.pStart + v;

	v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->hashValue = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void NestedClassTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x29 - NestedClass
	// "\x02*\x02*",

	tMD_NestedClass * p = (tMD_NestedClass*)pDest;
	memset(p, 0, sizeof(tMD_NestedClass));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->nestedClass = Coded2Index(pThis, 0x2, &pSource);

	p->enclosingClass = Coded2Index(pThis, 0x2, &pSource);

	*ppSource = pSource;
}

function_space_specifier void GenericParamTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x2A - Generic param
	// "s2s2<4S8"

	tMD_GenericParam * p = (tMD_GenericParam*)pDest;
	memset(p, 0, sizeof(tMD_GenericParam));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->number = GetU16(pSource);
	pSource += 2;

	p->flags = GetU16(pSource);
	pSource += 2;

	p->owner = CodedIndex(pThis, '<', &pSource);

	int string_skip = pThis->index32BitString ? 4 : 2;
	unsigned int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
	pSource += string_skip;
	p->name = (STRING)pThis->strings.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void MethodSpecTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x2B - MethodSpec
	// "x8m874B8"

	tMD_MethodSpec * p = (tMD_MethodSpec*)pDest;
	memset(p, 0, sizeof(tMD_MethodSpec));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->pMetaData = pThis;

	p->method = CodedIndex(pThis, '7', &pSource);

	int blob_skip = pThis->index32BitBlob ? 4 : 2;
	unsigned int v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
	pSource += blob_skip;
	p->instantiation = pThis->blobs.pStart + v;

	*ppSource = pSource;
}

function_space_specifier void GenericParamConstraintTableReader(tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int row, int numRows)
{
	// Table 0x2C - GenericParamConstraint
	// "\x2a*0*"

	tMD_GenericParamConstraint * p = (tMD_GenericParamConstraint*)pDest;
	memset(p, 0, sizeof(tMD_GenericParamConstraint));

	unsigned char * pSource = (unsigned char *)*ppSource;

	p->pGenericParam = (tMD_GenericParam *)Coded2Index(pThis, 0x2a, &pSource);

	p->constraint = CodedIndex(pThis, '0', &pSource);

	*ppSource = pSource;
}



// Loads a single table, returns pointer to table in memory.
function_space_specifier static void* LoadSingleTable(tMetaData *pThis, tRVA *pRVA, int tableID, void **ppTable) {
	int numRows = pThis->tables.numRows[tableID];
	int rowLen = 0; // Number of bytes taken by each row in memory.
	int row;
	const char *pDef = tableDefs[tableID];
	int defLen = (int)Gstrlen(pDef);
	void *pRet;
	unsigned char *pSource = (unsigned char*)*ppTable;
	char *pDest;


	// Set up row information of the metadata tables. With the structs defined
	// in MetaDataTables.h, this information is used to set up native access of
	// the metadata tables. Note, this is metadata format version specific.
	// Get destination size based on type.
	// Calculate the destination row size from table definition, if it hasn't already been calculated
	int newRowLen = 0;
	int numFields = defLen / 2;

	printf("TableID %x\n", tableID);

	switch (tableID)
	{
		case MD_TABLE_MODULE: rowLen = sizeof(tMD_Module); break;
		case MD_TABLE_TYPEREF: rowLen = sizeof(tMD_TypeRef); break;
		case MD_TABLE_TYPEDEF: rowLen = sizeof(tMD_TypeDef); break;
		case MD_TABLE_FIELDPTR: rowLen = sizeof(tMD_FieldPtr); break;
		case MD_TABLE_FIELDDEF: rowLen = sizeof(tMD_FieldDef); break;
		case MD_TABLE_METHODDEF: rowLen = sizeof(tMD_MethodDef); break;
		case MD_TABLE_PARAM: rowLen = sizeof(tMD_Param); break;
		case MD_TABLE_INTERFACEIMPL: rowLen = sizeof(tMD_InterfaceImpl); break;
		case MD_TABLE_MEMBERREF: rowLen = sizeof(tMD_MemberRef); break;
		case MD_TABLE_CONSTANT: rowLen = sizeof(tMD_Constant); break;
		case MD_TABLE_CUSTOMATTRIBUTE: rowLen = sizeof(tMD_CustomAttribute); break;
		case MD_TABLE_DECLSECURITY: rowLen = sizeof(tMD_DeclSecurity); break;
		case MD_TABLE_CLASSLAYOUT: rowLen = sizeof(tMD_ClassLayout); break;
		case MD_TABLE_STANDALONESIG: rowLen = sizeof(tMD_StandAloneSig); break;
		case MD_TABLE_EVENTMAP: rowLen = sizeof(tMD_EventMap); break;
		case MD_TABLE_EVENT: rowLen = sizeof(tMD_Event); break;
		case MD_TABLE_PROPERTYMAP: rowLen = sizeof(tMD_PropertyMap); break;
		case MD_TABLE_PROPERTY: rowLen = sizeof(tMD_Property); break;
		case MD_TABLE_METHODSEMANTICS: rowLen = sizeof(tMD_MethodSemantics); break;
		case MD_TABLE_METHODIMPL: rowLen = sizeof(tMD_MethodImpl); break;
		case MD_TABLE_MODULEREF: rowLen = sizeof(tMD_ModuleRef); break;
		case MD_TABLE_TYPESPEC: rowLen = sizeof(tMD_TypeSpec); break;
		case MD_TABLE_IMPLMAP: rowLen = sizeof(tMD_ImplMap); break;
		case MD_TABLE_FIELDRVA: rowLen = sizeof(tMD_FieldRVA); break;
		case MD_TABLE_ASSEMBLY: rowLen = sizeof(tMD_Assembly); break;
		case MD_TABLE_ASSEMBLYREF: rowLen = sizeof(tMD_AssemblyRef); break;
		case MD_TABLE_NESTEDCLASS: rowLen = sizeof(tMD_NestedClass); break;
		case MD_TABLE_GENERICPARAM: rowLen = sizeof(tMD_GenericParam); break;
		case MD_TABLE_METHODSPEC: rowLen = sizeof(tMD_MethodSpec); break;
		case MD_TABLE_GENERICPARAMCONSTRAINT: rowLen = sizeof(tMD_GenericParamConstraint); break;
	}
	tableRowSize[tableID] = rowLen;
	
	// Stuff fields described by pDef into appropriate table type. All types defined in MetaData.h

	// Allocate memory for destination table
	pRet = Gmalloc(numRows * rowLen);
	pThis->tables.data[tableID] = pRet;
	pDest = (char*) pRet;

	// Load rows of table, of give type of table.
	for (row=0; row<numRows; row++) {

		switch (tableID)
		{
			case MD_TABLE_MODULE: ModuleTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row))); continue;
			case MD_TABLE_TYPEREF: TypeRefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row))); continue;
			case MD_TABLE_TYPEDEF: TypeDefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_FIELDPTR: FieldPtrTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_FIELDDEF: FieldDefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_METHODDEF: MethodDefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_PARAM: ParamTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_INTERFACEIMPL: InterfaceImplTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_MEMBERREF: MemberRefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_CONSTANT: ConstantTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_CUSTOMATTRIBUTE: CustomAttributeTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_DECLSECURITY: DeclSecurityTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_CLASSLAYOUT: ClassLayoutTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_STANDALONESIG: StandAloneSigTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_EVENTMAP: EventMapTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_EVENT: EventTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_PROPERTYMAP: PropertyMapTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_PROPERTY: PropertyTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_METHODSEMANTICS: MethodSemanticsTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_METHODIMPL: MethodImplTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_MODULEREF: ModuleRefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_TYPESPEC: TypeSpecTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_IMPLMAP: ImplMapTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_FIELDRVA: FieldRVATableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_ASSEMBLY: AssemblyTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_ASSEMBLYREF: AssemblyRefTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_NESTEDCLASS: NestedClassTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_GENERICPARAM: GenericParamTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_METHODSPEC: MethodSpecTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;
			case MD_TABLE_GENERICPARAMCONSTRAINT: GenericParamConstraintTableReader(pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), row, numRows); continue;

			default: break;
		}
	}

	// Sanity check while debugging....
	for (row = 0; row < numRows; row++)
	{
		void * dd = MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, row + 1));
		switch (tableID)
		{
		case MD_TABLE_MODULE:
		{
			tMD_Module* y = (tMD_Module*)dd;
			printf("x1 %s\n", y->name);
			break;
		}
		case MD_TABLE_TYPEREF:
		{
			tMD_TypeRef* y = (tMD_TypeRef*)dd;
			printf("x2 %s\n", y->name);
			break;
		}
		case MD_TABLE_TYPEDEF:
		{
			tMD_TypeDef* y = (tMD_TypeDef*)dd;
			printf("x3 %s\n", y->name);
			break;
		}
		case MD_TABLE_FIELDDEF:
		{
			tMD_FieldDef* y = (tMD_FieldDef*)dd;
			break;
		}
		case MD_TABLE_METHODDEF:
		{
			tMD_MethodDef* y = (tMD_MethodDef*)dd;
			break;
		}
		case MD_TABLE_PARAM:
		{
			tMD_Param* y = (tMD_Param*)dd;
			break;
		}
		case MD_TABLE_INTERFACEIMPL:
		{
			tMD_InterfaceImpl* y = (tMD_InterfaceImpl*)dd;
			break;
		}
		case MD_TABLE_MEMBERREF:
		{
			tMD_MemberRef* y = (tMD_MemberRef*)dd;
			break;
		}
		case MD_TABLE_CONSTANT:
		{
			tMD_Constant* y = (tMD_Constant*)dd;
			break;
		}
		case MD_TABLE_CUSTOMATTRIBUTE:
		{
			tMD_CustomAttribute* y = (tMD_CustomAttribute*)dd;
			break;
		}
		case MD_TABLE_DECLSECURITY:
		{
			tMD_DeclSecurity* y = (tMD_DeclSecurity*)dd;
			break;
		}
		case MD_TABLE_CLASSLAYOUT:
		{
			tMD_ClassLayout* y = (tMD_ClassLayout*)dd;
			break;
		}
		case MD_TABLE_STANDALONESIG:
		{
			tMD_StandAloneSig* y = (tMD_StandAloneSig*)dd;
			break;
		}
		case MD_TABLE_EVENTMAP:
		{
			tMD_EventMap* y = (tMD_EventMap*)dd;
			break;
		}
		case MD_TABLE_EVENT:
		{
			tMD_Event* y = (tMD_Event*)dd;
			break;
		}
		case MD_TABLE_PROPERTYMAP:
		{
			tMD_PropertyMap* y = (tMD_PropertyMap*)dd;
			break;
		}
		case MD_TABLE_PROPERTY:
		{
			tMD_Property* y = (tMD_Property*)dd;
			break;
		}
		case MD_TABLE_METHODSEMANTICS:
		{
			tMD_MethodSemantics* y = (tMD_MethodSemantics*)dd;
			break;
		}
		case MD_TABLE_METHODIMPL:
		{
			tMD_MethodImpl* y = (tMD_MethodImpl*)dd;
			break;
		}
		case MD_TABLE_MODULEREF:
		{
			tMD_ModuleRef* y = (tMD_ModuleRef*)dd;
			break;
		}
		default:
			break;
		}
	}

	log_f(1, "Loaded MetaData table 0x%02X; %d rows\n", tableID, numRows);

	// Update the parameter to the position after this table
	*ppTable = pSource;
	// Return new table information
	return pRet;
}

function_space_specifier void MetaData_LoadTables(tMetaData *pThis, tRVA *pRVA, unsigned char *pStream, unsigned int streamLen) {
	U64 valid, j;
	unsigned char c;
	int i, k, numTables;
	void *pTable;

	unsigned char * ps = pStream;
	for (int i = 0; i < 16; ++i)
		printf("%x\n", ps[i]);

	ps += 6;
	c = *ps;
	pThis->index32BitString = (c & 1) > 0;
	pThis->index32BitGUID = (c & 2) > 0;
	pThis->index32BitBlob = (c & 4) > 0;
	ps += 2;
	valid = GetU64(ps);
	printf("valid = %llx\n", valid);
	// Count how many tables there are, and read in all the number of rows of each table.
	numTables = 0;
	for (i=0, j=1; i<MAX_TABLES; i++, j <<= 1) {
		// "valid" is a bitmap indicating if the table entry is OK. There are maximum
		// 48 (MAX_TABLES), but only those with bit set is valid.
		if (valid & j) {
			U32 vvv = GetU32(&((unsigned char*)pStream)[24 + numTables * 4]);
			pThis->tables.numRows[i] = vvv;
			numTables++;
			printf("Row v = %d %d\n", i, vvv);
		} else {
			pThis->tables.numRows[i] = 0;
			pThis->tables.data[i] = NULL;
		}
	}

	printf("Num tables %d\n", numTables);

	// Determine if each coded index lookup type needs to use 16 or 32 bit indexes
	for (i=0; i<13; i++) {
		const char* pCoding = codedTags[i];
		int tagBits = codedTagBits[i];
		// Discover max table size
		unsigned int maxTableLen = 0;
		for (k=0; k < (1<<tagBits); k++) {
			unsigned char t = pCoding[k];
			if (t != 'z') {
				if (pThis->tables.numRows[t] > maxTableLen) {
					maxTableLen = pThis->tables.numRows[t];
				}
			}
		}
		if (maxTableLen < (unsigned)(1 << (16 - tagBits))) {
			// Use 16-bit number
			pThis->tables.codedIndex32Bit[i] = 0;
		} else {
			// Use 32-bit number
			pThis->tables.codedIndex32Bit[i] = 1;
		}
	}

	pTable = &((char*)pStream)[24 + numTables * 4];

	for (i=0; i<MAX_TABLES; i++) {
		if (pThis->tables.numRows[i] > 0) {

printf("i = %d\n", i);
			if (i*4 >= sizeof(tableDefs) || tableDefs[i] == NULL) {
				printf("No table definition for MetaData table 0x%02x\n", i);
				gpuexit(1);
			}
			pThis->tables.data[i] = LoadSingleTable(pThis, pRVA, i, &pTable);
		}
	}
	printf("tables done.\n");
}

function_space_specifier PTR MetaData_GetBlob(BLOB_ blob, U32 *pBlobLength) {
	unsigned int len = MetaData_DecodeHeapEntryLength(&blob);
	if (pBlobLength != NULL) {
		*pBlobLength = len;
	}
	return blob;
}

// Returns length in bytes, not characters
function_space_specifier STRING2 MetaData_GetUserString(tMetaData *pThis, IDX_USERSTRINGS index, unsigned int *pStringLength) {
	unsigned char *pString = pThis->userStrings.pStart + (index & 0x00ffffff);
	unsigned int len = MetaData_DecodeHeapEntryLength(&pString);
	if (pStringLength != NULL) {
		// -1 because of extra terminating character in the heap
		*pStringLength = len - 1;
	}
	return (STRING2)pString;
}

function_space_specifier void* MetaData_GetTableRow(tMetaData *pThis, IDX_TABLE index) {
	char *pData;
	
	if (TABLE_OFS(index) == 0) {
		return NULL;
	}
	int table_id = TABLE_ID(index);
	void * d = pThis->tables.data[table_id];
	pData = (char*)pThis->tables.data[TABLE_ID(index)];
	// Table indexes start at one, hence the -1 here.
	int size = tableRowSize[TABLE_ID(index)];
	char * result = pData + (TABLE_OFS(index) - 1) * size;
	return result;
}

function_space_specifier void MetaData_GetConstant(tMetaData *pThis, IDX_TABLE idx, PTR pResultMem) {
	tMD_Constant *pConst = 0;

	switch (TABLE_ID(idx)) {
	case MD_TABLE_FIELDDEF:
		{
			tMD_FieldDef *pField = (tMD_FieldDef*)MetaData_GetTableRow(pThis, idx);
			pConst = (tMD_Constant*)pField->pMemory;
		}
		break;
	default:
		Crash("MetaData_GetConstant() Cannot handle idx: 0x%08x", idx);
	}

	switch (pConst->type) {
	case ELEMENT_TYPE_I4:
		//*(U32*)pReturnMem = MetaData_DecodeSigEntry(
		memcpy(pResultMem, pConst->value+1, 4);
		return;
	default:
		Crash("MetaData_GetConstant() Cannot handle value type: 0x%02x", pConst->type);
	}

}

function_space_specifier void MetaData_GetHeapRoots(tHeapRoots *pHeapRoots, tMetaData *pMetaData) {
	U32 i, top;
	// Go through all types, getting their static variables.

	top = pMetaData->tables.numRows[MD_TABLE_TYPEDEF];
	for (i=1; i<=top; i++) {
		tMD_TypeDef *pTypeDef;

		pTypeDef = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, MAKE_TABLE_INDEX(MD_TABLE_TYPEDEF, i));
		if (pTypeDef->isGenericDefinition) {
			Generic_GetHeapRoots(pHeapRoots, pTypeDef);
		} else {
			if (pTypeDef->staticFieldSize > 0) {
				Heap_SetRoots(pHeapRoots, pTypeDef->pStaticFields, pTypeDef->staticFieldSize);
			}
		}
	}
}