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

// Return the length of the signature. The length is an unsigned 32-bit integer
// of the number of bytes following the encoded length. Remember, the length
// is compressed, and can occupy more than one byte, so psig must be adjusted
// to be the first byte following the length.
function_space_specifier unsigned int MetaData_DecodeUnsigned32BitInteger(SIG *pSig)
{
    unsigned char a,b,c,d;
    unsigned char* ptr = ((unsigned char*)*pSig);
    a = *ptr++;
    *pSig = ptr;
    if ((a & 0x80) == 0) {
        // 1-byte entry
        return a;
    }
    // Special case
    if (a == 0xff) {
        return 0;
    }

    b = *ptr++;
    *pSig = ptr;
    if ((a & 0xc0) == 0x80) {
        // 2-byte entry
        return ((int)(a & 0x3f)) << 8 | b;
    }
    // 4-byte entry
    c = *ptr++;
    *pSig = ptr;
    d = *ptr++;
    *pSig = ptr;
    return ((int)(a & 0x1f)) << 24 | ((int)b) << 16 | ((int)c) << 8 | d;
}

function_space_specifier unsigned int MetaData_DecodeUnsigned8BitInteger(SIG *pSig) {
    unsigned char a, b, c, d;
    unsigned char* ptr = ((unsigned char*)*pSig);
    a = *ptr++;
    *pSig = ptr;
    return a;
}

function_space_specifier IDX_TABLE MetaData_DecodeSigEntryToken(SIG *pSig) {
    static U8 tableID[4] = {MD_TABLE_TYPEDEF, MD_TABLE_TYPEREF, MD_TABLE_TYPESPEC, 0};

    U32 entry = MetaData_DecodeUnsigned32BitInteger(pSig);
    return MAKE_TABLE_INDEX(tableID[entry & 0x3], entry >> 2);
}

function_space_specifier STRING MetaData_DecodePublicKey(BLOB_ blob)
{
    char * buf;
    unsigned int size = (unsigned int)blob[0];
    unsigned int buf_size = size * 2 + 1;
    buf = (char*)Gmalloc(buf_size);
    memset(buf, 0, buf_size);
    for (int i = 1; i <= size; ++i)
    {
        char hex[32];
        Gsprintf(hex, "%02x", blob[i]);
        Gstrcat(buf, hex);
    }
    return buf;
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
    return MetaData_DecodeUnsigned32BitInteger((SIG*)ppHeapEntry);
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

// function_space_specifier static unsigned int tableRowSize[MAX_TABLES];

function_space_specifier void MetaData_Init() {
    U32 i;
    for (i=0; i<MAX_TABLES; i++) {
        _bcl_->tableRowSize[i] = 0;
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
    char* codedTags[] = {
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
    unsigned char codedTagBits[] = {
        2, 2, 5, 1, 2, 3, 1, 1, 1, 2, 3, 2, 1
    };
    const char* pCoding = codedTags[ofs];
    int tagBits = codedTagBits[ofs];
    unsigned char tag = *pSource & ((1 << tagBits) - 1);
    int idxIntoTableID = pCoding[tag]; // The actual table index that we're looking for
    if (idxIntoTableID < 0 || idxIntoTableID > MAX_TABLES) {
        Crash("Error: Bad table index: 0x%02x\n", idxIntoTableID);
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

function_space_specifier void OutputSignature(unsigned char * ptr)
{
    // Get length.
    int length = 0;
    if (((*ptr) & 0x80) == 0)
    {
        length = *ptr;
        ptr++;
    }
    else if (((*ptr) & 0x40) == 0)
    {
        length = ((*ptr) & ~0x80) << 8;
        ptr++;
        length |= *ptr;
        ptr++;
    }
    else
    {
        length = ((*ptr) & ~0xc0) << 24;
        ptr++;
        length |= (*ptr) << 16;
        ptr++;
        length |= (*ptr) << 8;
        ptr++;
        length |= (*ptr);
        ptr++;
    }
    Gprintf("sig len   %d\n", length);
    Gprintf("sig data  ");
    // Now get data.
    for (int i = 0; i < length; ++i)
    {
        Gprintf("0x%02x ", *ptr++);
        if (i > 100) break;
    }
    Gprintf("\n");
}

function_space_specifier void ModuleTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest)
{
    // 0x00
    // original "sxS*G*GxGx",
    tMD_Module * p = (tMD_Module*)pDest;
    memset(p, 0, sizeof(tMD_Module));
    p->identity = MAKE_TABLE_INDEX(table, row);
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

function_space_specifier void OutputModule(tMD_Module* p)
{
    Gprintf("Module\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name %s\n", p->name);
    Gprintf("\n");
}

function_space_specifier void TypeRefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest)
{
    // 0x01
    // original "x*;*S*S*",
    tMD_TypeRef * p = (tMD_TypeRef*)pDest;
    memset(p, 0, sizeof(tMD_TypeRef));
    p->identity = MAKE_TABLE_INDEX(table, row);
    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    v = CodedIndex(pThis, ';', &pSource);
    p->resolutionScope = v;

    int string_skip = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->name_offset = v;
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->nameSpace_offset = v;
    pSource += string_skip;
    p->nameSpace = (STRING)pThis->strings.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputTypeRef(tMD_TypeRef * p)
{
    Gprintf("TypeRef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name_offset 0x%x\n", p->name_offset);
    Gprintf("name %s\n", p->name);
    Gprintf("nameSpace_offset 0x%x\n", p->nameSpace_offset);
    Gprintf("nameSpace %s\n", p->nameSpace);
    Gprintf("resolution 0x%08x\n", p->resolutionScope);
    Gprintf("\n");
}

function_space_specifier void TypeDefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
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
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->flags = GetU32(pSource);
    pSource += 4;

    int string_skip = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->name_offset = v;
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->nameSpace_offset = v;
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

function_space_specifier void OutputTypeDef(tMD_TypeDef * p)
{
    Gprintf("TypeDef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name_offset 0x%x\n", p->name_offset);
    Gprintf("name %s\n", p->name);
    Gprintf("nameSpace_offset 0x%x\n", p->nameSpace_offset);
    Gprintf("nameSpace %s\n", p->nameSpace);
    Gprintf("flags %ld\n", p->flags);
    Gprintf("extends %ld\n", p->extends);
    Gprintf("\n");
}

function_space_specifier void FieldPtrTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    tMD_FieldPtr * p = (tMD_FieldPtr*)pDest;
    memset(p, 0, sizeof(tMD_FieldPtr));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;
    // Missing???

    *ppSource = pSource;
}

function_space_specifier void OutputFieldPtr(tMD_FieldPtr * p)
{
    Gprintf("FieldPtr\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("\n");
}

function_space_specifier void FieldDefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // "x*m*ssxsS*B*x*x*x*x*I*x*",

    tMD_FieldDef * p = (tMD_FieldDef*)pDest;
    memset(p, 0, sizeof(tMD_FieldDef));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->flags = GetU16(pSource);
    pSource += 2;

    int string_skip = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->name_offset = v;
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->signature_offset = v;
    pSource += blob_skip;
    p->signature = pThis->blobs.pStart + v;

    p->pMetaData = pThis;

    p->tableIndex = MAKE_TABLE_INDEX(MD_TABLE_FIELDDEF, row + 1);

    *ppSource = pSource;
}

function_space_specifier void OutputFieldDef(tMD_FieldDef * p)
{
    Gprintf("FieldDef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("flags 0x%x\n", p->flags);
    Gprintf("name_offset 0x%x\n", p->name_offset);
    Gprintf("name %s\n", p->name);
    Gprintf("signature_offset 0x%x\n", p->signature_offset);
    OutputSignature(p->signature);
    Gprintf("pMetaData 0x%llx\n", p->pMetaData);
    Gprintf("tableIndx 0x%x\n", p->tableIndex);
    Gprintf("\n");
}

function_space_specifier void MethodDefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    //  "x*m*^*ssssS*B*\x08*x*x*x*x*x*x*I*x*x*x*"

    tMD_MethodDef * p = (tMD_MethodDef*)pDest;
    memset(p, 0, sizeof(tMD_MethodDef));
    p->identity = MAKE_TABLE_INDEX(table, row);

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
    p->name_offset = v;
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->signature_offset = v;
    pSource += blob_skip;
    p->signature = pThis->blobs.pStart + v;

    p->paramList = Coded2Index(pThis, 8, &pSource);;

    p->tableIndex = MAKE_TABLE_INDEX(MD_TABLE_METHODDEF, row + 1);

    *ppSource = pSource;
}

function_space_specifier void OutputMethodDef(tMD_MethodDef * p)
{
    Gprintf("MethodDef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name_offset %x\n", p->name_offset);
    Gprintf("name %s\n", p->name);
    Gprintf("flags %x\n", p->flags);
    Gprintf("ImplFlags %x\n", p->implFlags);
    Gprintf("signature_offset %x\n", p->signature_offset);
    OutputSignature(p->signature);
    Gprintf("\n");
}

function_space_specifier void ParamTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // 0x08
    // "ssssS*",

    tMD_Param * p = (tMD_Param*)pDest;
    memset(p, 0, sizeof(tMD_Param));
    p->identity = MAKE_TABLE_INDEX(table, row);

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
    p->name_offset = v;
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputParam(tMD_Param * p)
{
    Gprintf("Param\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name_offset %x\n", p->name_offset);
    Gprintf("name %s\n", p->name);
    Gprintf("flags %x\n", p->flags);
    Gprintf("sequence %x\n", p->sequence);
    Gprintf("\n");
}

function_space_specifier void InterfaceImplTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x09 - InterfaceImpl
    // "\x02*0*",

    tMD_InterfaceImpl * p = (tMD_InterfaceImpl*)pDest;
    memset(p, 0, sizeof(tMD_InterfaceImpl));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->class_ = Coded2Index(pThis, 4, &pSource);

    p->interface_ = CodedIndex(pThis, '0', &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputInterfaceImpl(tMD_InterfaceImpl * p)
{
    Gprintf("InterfaceImpl\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("class %x\n", p->class_);
    Gprintf("interface %x\n", p->interface_);
    Gprintf("\n");
}

function_space_specifier void MemberRefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x0A - MemberRef
    // "x*5*S*B*",

    tMD_MemberRef * p = (tMD_MemberRef*)pDest;
    memset(p, 0, sizeof(tMD_MemberRef));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->class_ = CodedIndex(pThis, '5', &pSource);

    int string_skip = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->name_offset = v;
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->signature_offset = v;
    pSource += blob_skip;
    p->signature = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputMemberRef(tMD_MemberRef * p)
{
    Gprintf("MemberRef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name_offset %x\n", p->name_offset);
    Gprintf("name %s\n", p->name);
    Gprintf("class %x\n", p->class_);
    Gprintf("signature_offset %x\n", p->signature_offset);
    OutputSignature(p->signature);
    Gprintf("\n");
}

function_space_specifier void ConstantTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x0B - Constant
    // "ccccxs1*B*",

    tMD_Constant * p = (tMD_Constant*)pDest;
    memset(p, 0, sizeof(tMD_Constant));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->type = *(U8*)pSource;
    pSource++;

    pSource++; // Skip intensional.

    p->parent = CodedIndex(pThis, '1', &pSource);

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->value_offset = v;
    pSource += blob_skip;
    p->value = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputConstant(tMD_Constant * p)
{
    Gprintf("Constant\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("type %x\n", p->type);
    Gprintf("value_offset %x\n", p->value_offset);
    Gprintf("parent %x\n", p->parent);
    Gprintf("\n");
}

function_space_specifier void CustomAttributeTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x0C - CustomAttribute
    // "2*:*B*",

    tMD_CustomAttribute * p = (tMD_CustomAttribute*)pDest;
    memset(p, 0, sizeof(tMD_CustomAttribute));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->parent = CodedIndex(pThis, '2', &pSource);

    p->type = CodedIndex(pThis, ':', &pSource);

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->value_offset = v;
    pSource += blob_skip;
    p->value = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputCustomAttribute(tMD_CustomAttribute * p)
{
    Gprintf("CustomAttribute\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("type %x\n", p->type);
    Gprintf("parent %x\n", p->parent);
    Gprintf("\n");
}

function_space_specifier void FieldMarshalTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x0d - FieldMarshal

    tMD_FieldMarshal * p = (tMD_FieldMarshal*)pDest;
    memset(p, 0, sizeof(tMD_FieldMarshal));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->type = CodedIndex(pThis, '3', &pSource);

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->value_offset = v;
    pSource += blob_skip;
    p->value = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputFieldMarshal(tMD_FieldMarshal * p)
{
    Gprintf("FieldMarshal\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("type %x\n", p->type);
    Gprintf("value_offset %x\n", p->value_offset);
    Gprintf("\n");
}

function_space_specifier void DeclSecurityTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x0E - DeclSecurity
    // "ssxs4*B*",

    tMD_DeclSecurity * p = (tMD_DeclSecurity*)pDest;
    memset(p, 0, sizeof(tMD_DeclSecurity));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    p->action = GetU16(pSource);
    pSource += 2;

    p->parent = CodedIndex(pThis, '4', &pSource);

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->permissionSet_offset = v;
    pSource += blob_skip;
    p->permissionSet = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputDeclSecurity(tMD_DeclSecurity * p)
{
    Gprintf("DeclSecurity\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("action %x\n", p->action);
    Gprintf("parent %x\n", p->parent);
    Gprintf("permissionSet_offset %x\n", p->permissionSet_offset);
    Gprintf("\n");
}

function_space_specifier void ClassLayoutTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x0F - ClassLayout
    // "ssxsi*\x02*",

    tMD_ClassLayout * p = (tMD_ClassLayout*)pDest;
    memset(p, 0, sizeof(tMD_ClassLayout));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->packingSize = GetU16(pSource);
    pSource += 2;

    p->classSize = GetU32(pSource);
    pSource += 4;

    p->parent = Coded2Index(pThis, 2, &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputClassLayout(tMD_ClassLayout * p)
{
    Gprintf("ClassLayout\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("packingSize %x\n", p->packingSize);
    Gprintf("classSize %x\n", p->classSize);
    Gprintf("parent %x\n", p->parent);
    Gprintf("\n");
}

function_space_specifier void FieldLayoutTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x10 - FieldLayout

    tMD_FieldLayout * p = (tMD_FieldLayout*)pDest;
    memset(p, 0, sizeof(tMD_FieldLayout));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->offset = GetU32(pSource);
    pSource += 4;

    p->field = Coded2Index(pThis, 4, &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputFieldLayout(tMD_FieldLayout * p)
{
    Gprintf("FieldLayout\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("offset %x\n", p->offset);
    Gprintf("field %x\n", p->field);
    Gprintf("\n");
}

function_space_specifier void StandAloneSigTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x11 - StandAloneSig
    // "B*",

    tMD_StandAloneSig * p = (tMD_StandAloneSig*)pDest;
    memset(p, 0, sizeof(tMD_StandAloneSig));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int v;

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->signature_offset = v;
    pSource += blob_skip;
    p->signature = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputStandAloneSig(tMD_StandAloneSig * p)
{
    Gprintf("StandAloneSig\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("signature_offset %x\n", p->signature_offset);
    OutputSignature(p->signature);
    Gprintf("\n");
}

function_space_specifier void EventMapTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x12 - EventMap
    // "\x02*\x14*",

    tMD_EventMap * p = (tMD_EventMap*)pDest;
    memset(p, 0, sizeof(tMD_EventMap));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->parent = Coded2Index(pThis, 2, &pSource);

    p->eventList = Coded2Index(pThis, 0x14, &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputEventMap(tMD_EventMap * p)
{
    Gprintf("EventMap\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("parent %x\n", p->parent);
    Gprintf("eventList %x\n", p->eventList);
    Gprintf("\n");
}

function_space_specifier void EventTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x14 - Event
    // "ssxsS*0*",

    tMD_Event * p = (tMD_Event*)pDest;
    memset(p, 0, sizeof(tMD_Event));
    p->identity = MAKE_TABLE_INDEX(table, row);

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

function_space_specifier void OutputEvent(tMD_Event * p)
{
    Gprintf("Event\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("eventFlags %x\n", p->eventFlags);
    Gprintf("name %s\n", p->name);
    Gprintf("eventType %x\n", p->eventType);
    Gprintf("\n");
}

function_space_specifier void PropertyMapTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x15 - PropertyMap
    // "\x02*\x17*",

    tMD_PropertyMap * p = (tMD_PropertyMap*)pDest;
    memset(p, 0, sizeof(tMD_PropertyMap));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->parent = Coded2Index(pThis, 0x2, &pSource);

    p->propertyList = Coded2Index(pThis, 0x17, &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputPropertyMap(tMD_PropertyMap * p)
{
    Gprintf("PropertyMap\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("parent %x\n", p->parent);
    Gprintf("propertyList %x\n", p->propertyList);
    Gprintf("\n");
}

function_space_specifier void PropertyTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x17 - Property
    // "ssxsS*B*",

    tMD_Property * p = (tMD_Property*)pDest;
    memset(p, 0, sizeof(tMD_Property));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->flags = GetU16(pSource);
    pSource += 2;

    int string_skip = pThis->index32BitString ? 4 : 2;
    int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->typeSig_offset = v;
    pSource += blob_skip;
    p->typeSig = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputProperty(tMD_Property * p)
{
    Gprintf("Property\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("flags %x\n", p->flags);
    Gprintf("name %s\n", p->name);
    Gprintf("\n");
}

function_space_specifier void MethodSemanticsTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x18 - MethodSemantics
    // "ssxs\06*6*",

    tMD_MethodSemantics * p = (tMD_MethodSemantics*)pDest;
    memset(p, 0, sizeof(tMD_MethodSemantics));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->semantics = GetU16(pSource);
    pSource += 2;

    p->method = Coded2Index(pThis, 0x6, &pSource);

    p->association = CodedIndex(pThis, '6', &pSource);
    *ppSource = pSource;
}

function_space_specifier void OutputMethodSemantics(tMD_MethodSemantics * p)
{
    Gprintf("MethodSemantics\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("semantics %x\n", p->semantics);
    Gprintf("method %x\n", p->method);
    Gprintf("association %x\n", p->association);
    Gprintf("\n");
}

function_space_specifier void MethodImplTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x19 - MethodImpl
    //  "\x02*7*7*"

    tMD_MethodImpl * p = (tMD_MethodImpl*)pDest;
    memset(p, 0, sizeof(tMD_MethodImpl));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->class_ = Coded2Index(pThis, 0x2, &pSource);

    p->methodBody = CodedIndex(pThis, '7', &pSource);

    p->methodDeclaration = CodedIndex(pThis, '7', &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputMethodImpl(tMD_MethodImpl * p)
{
    Gprintf("MethodImpl\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("class %x\n", p->class_);
    Gprintf("methodBody %x\n", p->methodBody);
    Gprintf("methodDeclaration %x\n", p->methodDeclaration);
    Gprintf("\n");
}

function_space_specifier void ModuleRefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x1a - ModuleRef
    //  "S*"

    tMD_ModuleRef * p = (tMD_ModuleRef*)pDest;
    memset(p, 0, sizeof(tMD_ModuleRef));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int string_skip = pThis->index32BitString ? 4 : 2;
    unsigned int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputModuleRef(tMD_ModuleRef * p)
{
    Gprintf("ModuleRef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name %s\n", p->name);
    Gprintf("\n");
}

function_space_specifier void TypeSpecTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x1B - TypeSpec
    //  "x*m*B*"

    tMD_TypeSpec * p = (tMD_TypeSpec*)pDest;
    memset(p, 0, sizeof(tMD_TypeSpec));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    unsigned int v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->signature_offset = v;
    pSource += blob_skip;
    p->signature = pThis->blobs.pStart + v;

    p->pMetaData = pThis;

    *ppSource = pSource;
}

function_space_specifier void OutputTypeSpec(tMD_TypeSpec * p)
{
    Gprintf("TypeSpec\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("typedef %llx\n", p->pTypeDef);
    OutputSignature(p->signature);
    Gprintf("\n");
}

function_space_specifier void ImplMapTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x1c - ImplMap
    //  "ssxs8*S*\x1a*"

    tMD_ImplMap * p = (tMD_ImplMap*)pDest;
    memset(p, 0, sizeof(tMD_ImplMap));
    p->identity = MAKE_TABLE_INDEX(table, row);

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

function_space_specifier void OutputImplMap(tMD_ImplMap * p)
{
    Gprintf("ImplMap\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("mappingFlags %x\n", p->mappingFlags);
    Gprintf("memberForwarded %x\n", p->memberForwarded);
    Gprintf("importName %s\n", p->importName);
    Gprintf("importScope %x\n", p->importScope);
    Gprintf("\n");
}

function_space_specifier void FieldRVATableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x1D - FieldRVA
    // "^*\x04*"

    tMD_FieldRVA * p = (tMD_FieldRVA*)pDest;
    memset(p, 0, sizeof(tMD_FieldRVA));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    unsigned int v = GetU32(pSource);
    pSource += 4;
    p->rva = RVA_FindData(pRVA, v);

    p->field = Coded2Index(pThis, 0x04, &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputFieldRVA(tMD_FieldRVA * p)
{
    Gprintf("FieldRVA\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("rva %llx\n", p->rva);
    Gprintf("field %x\n", p->field);
    Gprintf("\n");
}

function_space_specifier void AssemblyTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x20 - Assembly
    // "i4s2s2s2s2i4B8S8S8"

    tMD_Assembly * p = (tMD_Assembly*)pDest;
    memset(p, 0, sizeof(tMD_Assembly));
    p->identity = MAKE_TABLE_INDEX(table, row);

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
    p->publicKey_offset = v;
    pSource += blob_skip;
    p->publicKey = pThis->blobs.pStart + v;
    p->public_key_str = MetaData_DecodePublicKey(p->publicKey);

    int string_skip = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->culture = (STRING)pThis->strings.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputAssembly(tMD_Assembly * p)
{
    Gprintf("Assembly\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("hashAlgID %x\n", p->hashAlgID);
    Gprintf("majorVersion %x\n", p->majorVersion);
    Gprintf("minorVersion %x\n", p->minorVersion);
    Gprintf("buildNumber %x\n", p->buildNumber);
    Gprintf("revisionNumber %x\n", p->revisionNumber);
    Gprintf("flags %x\n", p->flags);
    Gprintf("name %s\n", p->name);
    Gprintf("culture %s\n", p->culture);
    Gprintf("\n");
}

function_space_specifier void AssemblyRefTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x23 - AssemblyRef
    // "s2s2s2s2i4B8S8S8B8"

    tMD_AssemblyRef * p = (tMD_AssemblyRef*)pDest;
    memset(p, 0, sizeof(tMD_AssemblyRef));
    p->identity = MAKE_TABLE_INDEX(table, row);

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
    p->publicKeyOrToken_offset = v;
    pSource += blob_skip;
    p->publicKeyOrToken = pThis->blobs.pStart + v;
    p->public_key_str = MetaData_DecodePublicKey(p->publicKeyOrToken);

    int string_skip = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->culture = (STRING)pThis->strings.pStart + v;

    v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->hashValue_offset = v;
    pSource += blob_skip;
    p->hashValue = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputAssemblyRef(tMD_AssemblyRef * p)
{
    Gprintf("AssemblyRef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("majorVersion %x\n", p->majorVersion);
    Gprintf("minorVersion %x\n", p->minorVersion);
    Gprintf("buildNumber %x\n", p->buildNumber);
    Gprintf("revisionNumber %x\n", p->revisionNumber);
    Gprintf("flags %x\n", p->flags);
    Gprintf("name %s\n", p->name);
    Gprintf("culture %s\n", p->culture);
    Gprintf("\n");
}

function_space_specifier void ExportedTypeTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x27

    tMD_ExportedType * p = (tMD_ExportedType*)pDest;
    memset(p, 0, sizeof(tMD_ExportedType));
    unsigned char * pSource = (unsigned char *)*ppSource;

    p->identity = MAKE_TABLE_INDEX(table, row);

    p->flags = GetU32(pSource);
    pSource += 4;

    p->TypeDefId = GetU32(pSource);
    pSource += 4;

    int string_skip = pThis->index32BitString ? 4 : 2;
    int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->offset_TypeName = v;
    pSource += string_skip;
    p->TypeName = (STRING)pThis->strings.pStart + v;

    int string_skip2 = pThis->index32BitString ? 4 : 2;
    v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    p->offset_TypeNamespace = v;
    pSource += string_skip2;
    p->TypeNamespace = (STRING)pThis->strings.pStart + v;

    p->Implementation = GetU16(pSource);
    pSource += 2;

    *ppSource = pSource;
}

function_space_specifier void OutputExportedType(tMD_ExportedType * p)
{
    Gprintf("AssemblyRef\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("flags %x\n", p->flags);
    Gprintf("TypeDefId 0x%04x\n", p->TypeDefId);
    Gprintf("TypeName %s\n", p->TypeName);
    Gprintf("TypeNamespace %s\n", p->TypeNamespace);
    Gprintf("Implementation 0x%04x\n", p->Implementation);
    Gprintf("\n");
}

function_space_specifier void ManifestResourceTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x28
    tMD_ManifestResource * p = (tMD_ManifestResource*)pDest;
    memset(p, 0, sizeof(tMD_ManifestResource));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->offset = GetU32(pSource);
    pSource += 4;

    p->flags = GetU32(pSource);
    pSource += 4;

    int string_skip = pThis->index32BitString ? 4 : 2;
    unsigned int v = pThis->index32BitString ? GetU32(pSource) : GetU16(pSource);
    pSource += string_skip;
    p->name = (STRING)pThis->strings.pStart + v;

    p->implementation = GetU16(pSource);
    pSource += 2;

    *ppSource = pSource;
}

function_space_specifier void OutputManifestResource(tMD_ManifestResource * p)
{
    Gprintf("ManifestResource\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("name %s\n", p->name);
    Gprintf("\n");
}

function_space_specifier void NestedClassTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x29 - NestedClass
    // "\x02*\x02*",

    tMD_NestedClass * p = (tMD_NestedClass*)pDest;
    memset(p, 0, sizeof(tMD_NestedClass));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->nestedClass = Coded2Index(pThis, 0x2, &pSource);

    p->enclosingClass = Coded2Index(pThis, 0x2, &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputNestedClass(tMD_NestedClass * p)
{
    Gprintf("NestedClass\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("nestedClass 0x%08x\n", p->nestedClass);
    Gprintf("enclosingClass 0x%08x\n", p->enclosingClass);
    Gprintf("\n");
}

function_space_specifier void GenericParamTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x2A - Generic param
    // "s2s2<4S8"

    tMD_GenericParam * p = (tMD_GenericParam*)pDest;
    memset(p, 0, sizeof(tMD_GenericParam));
    p->identity = MAKE_TABLE_INDEX(table, row);

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

function_space_specifier void OutputGenericParam(tMD_GenericParam * p)
{
    Gprintf("GenericParam\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("number %x\n", p->number);
    Gprintf("flags %x\n", p->flags);
    Gprintf("owner %x\n", p->owner);
    Gprintf("name %x\n", p->name);
    Gprintf("\n");
}

function_space_specifier void MethodSpecTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x2B - MethodSpec
    // "x8m874B8"

    tMD_MethodSpec * p = (tMD_MethodSpec*)pDest;
    memset(p, 0, sizeof(tMD_MethodSpec));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->pMetaData = pThis;

    p->method = CodedIndex(pThis, '7', &pSource);

    int blob_skip = pThis->index32BitBlob ? 4 : 2;
    unsigned int v = pThis->index32BitBlob ? GetU32(pSource) : GetU16(pSource);
    p->instantiation_offset = v;
    pSource += blob_skip;
    p->instantiation = pThis->blobs.pStart + v;

    *ppSource = pSource;
}

function_space_specifier void OutputMethodSpec(tMD_MethodSpec * p)
{
    Gprintf("MethodSpec\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("method %x\n", p->method);
    Gprintf("\n");
}

function_space_specifier void GenericParamConstraintTableReader(int table, int row, tMetaData *pThis, tRVA *pRVA, unsigned char **ppSource, void *pDest, int numRows)
{
    // Table 0x2C - GenericParamConstraint
    // "\x2a*0*"

    tMD_GenericParamConstraint * p = (tMD_GenericParamConstraint*)pDest;
    memset(p, 0, sizeof(tMD_GenericParamConstraint));
    p->identity = MAKE_TABLE_INDEX(table, row);

    unsigned char * pSource = (unsigned char *)*ppSource;

    p->pGenericParam = (tMD_GenericParam *)Coded2Index(pThis, 0x2a, &pSource);

    p->constraint = CodedIndex(pThis, '0', &pSource);

    *ppSource = pSource;
}

function_space_specifier void OutputGenericParamConstraint(tMD_GenericParamConstraint * p)
{
    Gprintf("GenericParamConstraint\n");
    Gprintf("id 0x%08x\n", p->identity);
    Gprintf("constraint %x\n", p->constraint);
    Gprintf("\n");
}

function_space_specifier static void* LoadSingleTable(tMetaData *pThis, tRVA *pRVA, int tableID, void **ppTable) {
    int numRows = pThis->tables.numRows[tableID];
    int rowLen = 0; // Number of bytes taken by each row in memory.
    int row;
    void *pRet;
    unsigned char *pSource = (unsigned char*)*ppTable;
    char *pDest;


    // Set up row information of the metadata tables. With the structs defined
    // in MetaDataTables.h, this information is used to set up native access of
    // the metadata tables. Note, this is metadata format version specific.
    // Get destination size based on type.
    // Calculate the destination row size from table definition, if it hasn't already been calculated
    int newRowLen = 0;

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
        case MD_TABLE_FIELDMARSHAL: rowLen = sizeof(tMD_FieldMarshal); break;
        case MD_TABLE_DECLSECURITY: rowLen = sizeof(tMD_DeclSecurity); break;
        case MD_TABLE_CLASSLAYOUT: rowLen = sizeof(tMD_ClassLayout); break;
        case MD_TABLE_FIELDLAYOUT: rowLen = sizeof(tMD_FieldLayout); break;
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
        case MD_TABLE_EXPORTEDTYPE: rowLen = sizeof(tMD_ExportedType); break;
        case MD_TABLE_MANIFESTRESOURCE: rowLen = sizeof(tMD_ManifestResource); break;
        case MD_TABLE_NESTEDCLASS: rowLen = sizeof(tMD_NestedClass); break;
        case MD_TABLE_GENERICPARAM: rowLen = sizeof(tMD_GenericParam); break;
        case MD_TABLE_METHODSPEC: rowLen = sizeof(tMD_MethodSpec); break;
        case MD_TABLE_GENERICPARAMCONSTRAINT: rowLen = sizeof(tMD_GenericParamConstraint); break;
        default:
            Crash("Unknow PE metadata table type %d.\n", tableID);
            break;
    }
    _bcl_->tableRowSize[tableID] = rowLen;
    
    // Stuff fields described by pDef into appropriate table type. All types defined in MetaData.h

    // Allocate memory for destination table
    pRet = Gmalloc(numRows * rowLen);
    if (pRet == NULL)
        Crash("Out of memory!\n");

    pThis->tables.data[tableID] = pRet;
    pDest = (char*) pRet;

    // Load rows of table, of give type of table.
    for (row=0; row<numRows; row++) {

        switch (tableID)
        {
            case MD_TABLE_MODULE: ModuleTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row))); continue;
            case MD_TABLE_TYPEREF: TypeRefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row))); continue;
            case MD_TABLE_TYPEDEF: TypeDefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_FIELDPTR: FieldPtrTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_FIELDDEF: FieldDefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_METHODDEF: MethodDefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_PARAM: ParamTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_INTERFACEIMPL: InterfaceImplTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_MEMBERREF: MemberRefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_CONSTANT: ConstantTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_CUSTOMATTRIBUTE: CustomAttributeTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_FIELDMARSHAL: FieldMarshalTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_DECLSECURITY: DeclSecurityTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_CLASSLAYOUT: ClassLayoutTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_FIELDLAYOUT: FieldLayoutTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_STANDALONESIG: StandAloneSigTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_EVENTMAP: EventMapTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_EVENT: EventTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_PROPERTYMAP: PropertyMapTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_PROPERTY: PropertyTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_METHODSEMANTICS: MethodSemanticsTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_METHODIMPL: MethodImplTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_MODULEREF: ModuleRefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_TYPESPEC: TypeSpecTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_IMPLMAP: ImplMapTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_FIELDRVA: FieldRVATableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_ASSEMBLY: AssemblyTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_ASSEMBLYREF: AssemblyRefTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_EXPORTEDTYPE: ExportedTypeTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_MANIFESTRESOURCE: ManifestResourceTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_NESTEDCLASS: NestedClassTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_GENERICPARAM: GenericParamTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_METHODSPEC: MethodSpecTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            case MD_TABLE_GENERICPARAMCONSTRAINT: GenericParamConstraintTableReader(tableID, row, pThis, pRVA, &pSource, MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, 1 + row)), numRows); continue;
            default:
                Crash("Unknow PE metadata table type %d.\n", tableID);
                break;
        }
    }

    // Sanity check while debugging....
    for (row = 0; row < numRows; row++)
    {
        void * dd = MetaData_GetTableRow(pThis, MAKE_TABLE_INDEX(tableID, row + 1));
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

    ps += 6;
    c = *ps;
    pThis->index32BitString = (c & 1) > 0;
    pThis->index32BitGUID = (c & 2) > 0;
    pThis->index32BitBlob = (c & 4) > 0;
    ps += 2;
    valid = GetU64(ps);
    // Count how many tables there are, and read in all the number of rows of each table.
    numTables = 0;
    for (i=0, j=1; i<MAX_TABLES; i++, j <<= 1) {
        // "valid" is a bitmap indicating if the table entry is OK. There are maximum
        // 48 (MAX_TABLES), but only those with bit set is valid.
        if (valid & j) {
            U32 vvv = GetU32(&((unsigned char*)pStream)[24 + numTables * 4]);
            pThis->tables.numRows[i] = vvv;
            numTables++;
        } else {
            pThis->tables.numRows[i] = 0;
            pThis->tables.data[i] = NULL;
        }
    }

    char* codedTags[] = {
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
    unsigned char codedTagBits[] = {
        2, 2, 5, 1, 2, 3, 1, 1, 1, 2, 3, 2, 1
    };

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
            pThis->tables.data[i] = LoadSingleTable(pThis, pRVA, i, &pTable);
        }
    }
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
    int size = _bcl_->tableRowSize[TABLE_ID(index)];
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

function_space_specifier void MetaData_PrintMetaData(tMetaData * meta)
{
    for (int i = 0; i<MAX_TABLES; i++) {
        if (meta->tables.numRows[i] > 0) {
            void * tables = meta->tables.data[i];
            int numRows = meta->tables.numRows[i];
            if (numRows == 0)
                continue;
            for (int row = 0; row < numRows; row++)
            {
                void * dd = MetaData_GetTableRow(meta, MAKE_TABLE_INDEX(i, row + 1));
                switch (i)
                {
                case MD_TABLE_MODULE:
                {
                    tMD_Module* y = (tMD_Module*)dd;
                    OutputModule(y);
                    break;
                }
                case MD_TABLE_TYPEREF:
                {
                    tMD_TypeRef* y = (tMD_TypeRef*)dd;
                    OutputTypeRef(y);
                    break;
                }
                case MD_TABLE_TYPEDEF:
                {
                    tMD_TypeDef* y = (tMD_TypeDef*)dd;
                    OutputTypeDef(y);
                    break;
                }
                case MD_TABLE_FIELDPTR:
                {
                    tMD_FieldPtr* y = (tMD_FieldPtr*)dd;
                    OutputFieldPtr(y);
                    break;
                }
                case MD_TABLE_FIELDDEF:
                {
                    tMD_FieldDef* y = (tMD_FieldDef*)dd;
                    OutputFieldDef(y);
                    break;
                }
                case MD_TABLE_METHODDEF:
                {
                    tMD_MethodDef* y = (tMD_MethodDef*)dd;
                    OutputMethodDef(y);
                    break;
                }
                case MD_TABLE_PARAM:
                {
                    tMD_Param* y = (tMD_Param*)dd;
                    OutputParam(y);
                    break;
                }
                case MD_TABLE_INTERFACEIMPL:
                {
                    tMD_InterfaceImpl* y = (tMD_InterfaceImpl*)dd;
                    OutputInterfaceImpl(y);
                    break;
                }
                case MD_TABLE_MEMBERREF:
                {
                    tMD_MemberRef* y = (tMD_MemberRef*)dd;
                    OutputMemberRef(y);
                    break;
                }
                case MD_TABLE_CONSTANT:
                {
                    tMD_Constant* y = (tMD_Constant*)dd;
                    OutputConstant(y);
                    break;
                }
                case MD_TABLE_CUSTOMATTRIBUTE:
                {
                    tMD_CustomAttribute* y = (tMD_CustomAttribute*)dd;
                    OutputCustomAttribute(y);
                    break;
                }
                case MD_TABLE_FIELDMARSHAL:
                {
                    tMD_FieldMarshal* y = (tMD_FieldMarshal*)dd;
                    OutputFieldMarshal(y);
                    break;
                }
                case MD_TABLE_DECLSECURITY:
                {
                    tMD_DeclSecurity* y = (tMD_DeclSecurity*)dd;
                    OutputDeclSecurity(y);
                    break;
                }
                case MD_TABLE_CLASSLAYOUT:
                {
                    tMD_ClassLayout* y = (tMD_ClassLayout*)dd;
                    OutputClassLayout(y);
                    break;
                }
                case MD_TABLE_FIELDLAYOUT:
                {
                    tMD_FieldLayout* y = (tMD_FieldLayout*)dd;
                    OutputFieldLayout(y);
                    break;
                }
                case MD_TABLE_STANDALONESIG:
                {
                    tMD_StandAloneSig* y = (tMD_StandAloneSig*)dd;
                    OutputStandAloneSig(y);
                    break;
                }
                case MD_TABLE_EVENTMAP:
                {
                    tMD_EventMap* y = (tMD_EventMap*)dd;
                    OutputEventMap(y);
                    break;
                }
                case MD_TABLE_EVENT:
                {
                    tMD_Event* y = (tMD_Event*)dd;
                    OutputEvent(y);
                    break;
                }
                case MD_TABLE_PROPERTYMAP:
                {
                    tMD_PropertyMap* y = (tMD_PropertyMap*)dd;
                    OutputPropertyMap(y);
                    break;
                }
                case MD_TABLE_PROPERTY:
                {
                    tMD_Property* y = (tMD_Property*)dd;
                    OutputProperty(y);
                    break;
                }
                case MD_TABLE_METHODSEMANTICS:
                {
                    tMD_MethodSemantics* y = (tMD_MethodSemantics*)dd;
                    OutputMethodSemantics(y);
                    break;
                }
                case MD_TABLE_METHODIMPL:
                {
                    tMD_MethodImpl* y = (tMD_MethodImpl*)dd;
                    OutputMethodImpl(y);
                    break;
                }
                case MD_TABLE_MODULEREF:
                {
                    tMD_ModuleRef* y = (tMD_ModuleRef*)dd;
                    OutputModuleRef(y);
                    break;
                }
                case MD_TABLE_TYPESPEC:
                {
                    tMD_TypeSpec* y = (tMD_TypeSpec*)dd;
                    OutputTypeSpec(y);
                    break;
                }
                case MD_TABLE_IMPLMAP:
                {
                    tMD_ImplMap* y = (tMD_ImplMap*)dd;
                    OutputImplMap(y);
                    break;
                }
                case MD_TABLE_FIELDRVA:
                {
                    tMD_FieldRVA* y = (tMD_FieldRVA*)dd;
                    OutputFieldRVA(y);
                    break;
                }
                case MD_TABLE_ASSEMBLY:
                {
                    tMD_Assembly* y = (tMD_Assembly*)dd;
                    OutputAssembly(y);
                    break;
                }
                case MD_TABLE_ASSEMBLYREF:
                {
                    tMD_AssemblyRef* y = (tMD_AssemblyRef*)dd;
                    OutputAssemblyRef(y);
                    break;
                }
                case MD_TABLE_EXPORTEDTYPE:
                {
                    tMD_ExportedType* y = (tMD_ExportedType*)dd;
                    OutputExportedType(y);
                    break;
                }
                case MD_TABLE_MANIFESTRESOURCE:
                {
                    tMD_ManifestResource* y = (tMD_ManifestResource*)dd;
                    OutputManifestResource(y);
                    break;
                }
                case MD_TABLE_NESTEDCLASS:
                {
                    tMD_NestedClass* y = (tMD_NestedClass*)dd;
                    OutputNestedClass(y);
                    break;
                }
                case MD_TABLE_GENERICPARAM:
                {
                    tMD_GenericParam* y = (tMD_GenericParam*)dd;
                    OutputGenericParam(y);
                    break;
                }
                case MD_TABLE_METHODSPEC:
                {
                    tMD_MethodSpec* y = (tMD_MethodSpec*)dd;
                    OutputMethodSpec(y);
                    break;
                }
                case MD_TABLE_GENERICPARAMCONSTRAINT:
                {
                    tMD_GenericParamConstraint* y = (tMD_GenericParamConstraint*)dd;
                    OutputGenericParamConstraint(y);
                    break;
                }
                default:
                    Crash("Unknow PE metadata table type %d.\n", i);
                    break;
                }
            }
        }
    }
}

function_space_specifier void * MetaData_GetField(HEAP_PTR object, tMD_FieldDef * pField)
{
    int field_offset = pField->memOffset;
    unsigned char * field = (unsigned char *)object + field_offset;
    int field_size = pField->memSize;
    return (void*)field;
}

function_space_specifier int MetaData_GetFieldSize(tMD_FieldDef * pField)
{
    int field_size = pField->memSize;
    return field_size;
}

function_space_specifier int MetaData_GetFieldOffset(tMD_FieldDef * pField)
{
    int field_offset = pField->memOffset;
    return field_offset;
}

function_space_specifier void * MetaData_GetStaticField(tMD_FieldDef * pField)
{
    if (pField->pMemory == NULL)
    {
        pField->pMemory = (PTR)Gmalloc(pField->memSize);
    }
    return (void*) pField->pMemory;
}

function_space_specifier void MetaData_GetFields(tMD_TypeDef * pTypeDef, tMD_FieldDef*** out_buf, int * out_len)
{
    *out_len = pTypeDef->numFields;
    *out_buf = pTypeDef->ppFields;
}

function_space_specifier void MetaData_GetFieldsAll(tMD_TypeDef * pTypeDef, tMD_FieldDef*** out_buf, int * out_len)
{
    *out_len = pTypeDef->numFieldsAll;
    *out_buf = pTypeDef->ppFieldsAll;
}

function_space_specifier char * MetaData_GetFieldName(tMD_FieldDef * pFieldDef)
{
    return pFieldDef->name;
}

function_space_specifier tMD_TypeDef * MetaData_GetFieldType(tMD_FieldDef * pFieldDef)
{
    return pFieldDef->pType;
}

function_space_specifier void * MetaData_GetMethodJit(void * object_ptr, int table_ref)
{
    tMD_TypeDef* type = Heap_GetType((HEAP_PTR)object_ptr);
    if (type == NULL) return NULL;

    tMetaData * pMetaData = type->pMetaData;

    tMD_MethodDef * pCallMethod = MetaData_GetMethodDefFromDefRefOrSpec(pMetaData, table_ref, NULL, NULL);

    if (pCallMethod->isFilled == 0) {
        tMD_TypeDef *pTypeDef;
        pTypeDef = MetaData_GetTypeDefFromMethodDef(pCallMethod);
        MetaData_Fill_TypeDef(pTypeDef, NULL, NULL);
    }

    int vtable_offset = pCallMethod->vTableOfs;
    if (vtable_offset >= 0)
    {
        tMD_MethodDef * vtable_entry = type->pVTable[vtable_offset];
        tJITted * ptJITted = vtable_entry->pJITted;
        if (ptJITted == NULL)
        {
            Gprintf("Warning--method %s JIT compiled object missing.\n", pCallMethod->name);
            return NULL;
        }
        void * call = (void *)ptJITted->code;
        call = (void*)(*(U64*)call);
        return call;
    }
    else
    {
        tJITted * ptJITted = pCallMethod->pJITted;
        if (ptJITted == NULL)
        {
            Gprintf("Warning--method %s JIT compiled object missing.\n", pCallMethod->name);
            return NULL;
        }
        void * call = (void *)ptJITted->code;
        call = (void*)(*(U64*)call);
        return call;
    }
}

function_space_specifier void MetaData_SetMethodJit(void * method_ptr, void * bcl_type, int token)
{
    tMD_TypeDef* type = (tMD_TypeDef*) bcl_type;
    if (type == NULL) return;

    tMetaData * pMetaData = type->pMetaData;
    if (MetaData_GetTableRow(pMetaData, token) == NULL)
        return;

    tMD_MethodDef * pCallMethod = MetaData_GetMethodDefFromDefRefOrSpec(pMetaData, token, NULL, NULL);
        
    if (pCallMethod->isFilled == 0) {
        tMD_TypeDef *pTypeDef;
        pTypeDef = MetaData_GetTypeDefFromMethodDef(pCallMethod);
        MetaData_Fill_TypeDef(pTypeDef, NULL, NULL);
    }

    int vtable_offset = pCallMethod->vTableOfs;
    if (vtable_offset >= 0)
    {
        tMD_MethodDef * vtable_entry = type->pVTable[vtable_offset];
        tJITted * ptJITted = vtable_entry->pJITted;
        if (ptJITted == NULL)
        {
            vtable_entry->pJITted = (tJITted*)Gmalloc(sizeof(tJITted));
            ptJITted = vtable_entry->pJITted;
        }
        ptJITted->code = method_ptr;
    }
    else
    {
        tJITted * ptJITted = pCallMethod->pJITted;
        if (ptJITted == NULL)
        {
            pCallMethod->pJITted = (tJITted*)Gmalloc(sizeof(tJITted));
            ptJITted = pCallMethod->pJITted;
        }
        ptJITted->code = method_ptr;
    }
}


