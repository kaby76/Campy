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

#include "_BCL_.h"
#include "Sys.h"
#include "CLIFile.h"
#include "RVA.h"
#include "MetaData.h"
#include "Thread.h"
#include "MetaDataTables.h"
#include "Type.h"
#include <stdio.h>

#include "System.Array.h"
#include "System.String.h"
#include "Gstring.h"
#include "Gprintf.h"
#include "Filesystem.h"

// Is this exe/dll file for the .NET virtual machine?
#define DOT_NET_MACHINE 0x14c

typedef struct tFilesLoaded_ tFilesLoaded;
struct tFilesLoaded_ {
	tCLIFile *pCLIFile;
	tFilesLoaded *pNext;
};

// Keep track of all the files currently loaded
//function_space_specifier static tFilesLoaded *pFilesLoaded = NULL;

__global__ void BCL_CLIFile_GetMetaDataForAssembly(char * fileName)
{
	tMetaData* result;
	result = CLIFile_GetMetaDataForAssembly(fileName);
}

function_space_specifier void CLIFile_PrintMetaData(tMetaData * meta)
{


}

function_space_specifier tMetaData* CLIFile_GetMetaDataForAssembly(char * fileName)
{
	tFilesLoaded *pFiles;
	char * pAssemblyName;
	char assemblyName[250];
	Gstrcpy(assemblyName, fileName);
	char * r = Gstrstr(assemblyName, ".exe");
	if (r > 0) *r = 0;
	r = Gstrstr(assemblyName, ".dll");
	if (r > 0) *r = 0;
	pAssemblyName = assemblyName;

	// Convert "mscorlib" to "corlib"
	if (Gstrcmp(pAssemblyName, "mscorlib") == 0) // Net Framework rewrite.
	{
		pAssemblyName = "corlib";
	}
	else if (Gstrcmp(pAssemblyName, "System.Runtime") == 0) // Net Core rewrite.
	{
		pAssemblyName = "corlib";
	}

	// Look in already-loaded files first
	pFiles = _bcl_->pFilesLoaded;
	while (pFiles != NULL) {
		tCLIFile *pCLIFile;
		tMD_Assembly *pThisAssembly;

		pCLIFile = pFiles->pCLIFile;
		// Get the assembly info - there is only ever one of these in the each file's metadata
		pThisAssembly = (tMD_Assembly*)MetaData_GetTableRow(pCLIFile->pMetaData, MAKE_TABLE_INDEX(0x20, 1));
		if (Gstrcmp(pAssemblyName, pThisAssembly->name) == 0) {
			// Found the correct assembly, so return its meta-data
			return pCLIFile->pMetaData;
		}
		pFiles = pFiles->pNext;
	}

	// Assembly not loaded, so load it if possible
	{
		tCLIFile *pCLIFile;
		pCLIFile = CLIFile_Load(fileName);
		//pCLIFile = CLIFile_Load(pAssemblyName);
		//printf("In CLIFile_GetMetaDataForAssembly2\n");
		if (pCLIFile == NULL) {
			Crash("Cannot load required assembly file: %s", fileName);
			return NULL;
		}
		return pCLIFile->pMetaData;
	}
}

// function_space_specifier unsigned char Gdata[];

function_space_specifier static void* LoadFileFromDisk(char *pFileName)
{
	char *pData = NULL;
	// Crashes! Gprintf("File name = %s\n", pFileName);
	int handle;
	Gfs_open_file(pFileName, &handle);
	Gfs_read(handle, &pData);

	//f = open(pFileName, O_RDONLY|O_BINARY);
	//if (f >= 0) {
	//	int len;
	//	len = lseek(f, 0, SEEK_END);
	//	lseek(f, 0, SEEK_SET);
	//	// TODO: Change to use mmap() or windows equivilent
	//	pData = mallocForever(len);
	//	if (pData != NULL) {
	//		int r = read(f, pData, len);
	//		if (r != len) {
	//			Gfree(pData);
	//			pData = NULL;
	//		}
	//	}
	//	close(f);
	//}
	//pData = Gdata;
	return pData;
}

function_space_specifier static tCLIFile* LoadPEFile(void *pData) {

	tCLIFile *pRet = TMALLOC(tCLIFile);
	memset(pRet, 0, sizeof(tCLIFile));

	unsigned char *pMSDOSHeader = (unsigned char*)&(((unsigned char*)pData)[0]);
	unsigned char *pPEHeader;
	unsigned char *pPEOptionalHeader;
	unsigned char *pPESectionHeaders;
	unsigned char *pCLIHeader;
	unsigned char *pRawMetaData;

	int i;
	unsigned int lfanew;
	unsigned short machine;
	int numSections;
	unsigned int imageBase;
	int fileAlignment;
	unsigned int cliHeaderRVA, cliHeaderSize;
	unsigned int metaDataRVA, metaDataSize;
	tMetaData *pMetaData;
	pRet->pRVA = RVA();
	pRet->pMetaData = pMetaData = MetaData();
	lfanew = GetU32(&(pMSDOSHeader[0x3c]));
	pPEHeader = pMSDOSHeader + lfanew + 4;
	pPEOptionalHeader = pPEHeader + 20;
	pPESectionHeaders = pPEOptionalHeader + 224;

	machine = GetU16(&(pPEHeader[0]));
	if (machine != DOT_NET_MACHINE) {
		Gprintf("Not DOT_NET_MACHINE.\n");
		return NULL;
	}

	numSections = GetU16(&(pPEHeader[2]));
	imageBase = GetU32(&(pPEOptionalHeader[28]));
	fileAlignment = GetU32(&(pPEOptionalHeader[36]));

	for (i=0; i<numSections; i++) {
		unsigned char *pSection = pPESectionHeaders + i * 40;
		RVA_Create(pRet->pRVA, pData, pSection);
	}

	cliHeaderRVA = GetU32(&(pPEOptionalHeader[208]));
	cliHeaderSize = GetU32(&(pPEOptionalHeader[212]));

	pCLIHeader = (unsigned char *)RVA_FindData(pRet->pRVA, cliHeaderRVA);

	metaDataRVA = GetU32(&(pCLIHeader[8]));
	metaDataSize = GetU32(&(pCLIHeader[12]));
	pRet->entryPoint = GetU32(&(pCLIHeader[20]));
	pRawMetaData = (unsigned char*)RVA_FindData(pRet->pRVA, metaDataRVA);

	// Load all metadata
	{
		unsigned int versionLen = GetU32(&(pRawMetaData[12]));
		unsigned int ofs, numberOfStreams;
		unsigned char *pTableStream = NULL;
		unsigned int tableStreamSize;
		pRet->pVersion = &(pRawMetaData[16]);
		log_f(1, "CLI version: %s\n", pRet->pVersion);
		ofs = 16 + versionLen;
		numberOfStreams = GetU16(&(pRawMetaData[ofs + 2]));
		ofs += 4;
		int q = ofs;

		for (i=0; i<(signed)numberOfStreams; i++) {
			// Start at ofs and look for '#Strings', '#US', etc. Backup to get offset and size.
			for (; ; ++q)
			{
				if (pRawMetaData[q] == '#')
				{
					if (pRawMetaData[q + 1] == 'S' && pRawMetaData[q + 2] == 't')
						break;
					if (pRawMetaData[q + 1] == 'U' && pRawMetaData[q + 2] == 'S')
						break;
					if (pRawMetaData[q + 1] == 'B' && pRawMetaData[q + 2] == 'l')
						break;
					if (pRawMetaData[q + 1] == 'G' && pRawMetaData[q + 2] == 'U')
						break;
					if (pRawMetaData[q + 1] == '~')
						break;
				}
			}
			ofs = q - 8;
			unsigned int streamOffset = *(unsigned int*)&pRawMetaData[ofs];
			unsigned int streamSize = *(unsigned int*)&pRawMetaData[ofs+4];
			unsigned char *pStreamName = &pRawMetaData[ofs+8];
			unsigned char *pStream = pRawMetaData + streamOffset;
			q = q + 1;
			//ofs += (unsigned int)((Gstrlen((const char*)pStreamName)+4) & (~0x3)) + 8;
			if (Gstrcasecmp((const char*)pStreamName, "#Strings") == 0) {
				MetaData_LoadStrings(pMetaData, pStream, streamSize);
			} else if (Gstrcasecmp((const char*)pStreamName, "#US") == 0) {
				MetaData_LoadUserStrings(pMetaData, pStream, streamSize);
			} else if (Gstrcasecmp((const char*)pStreamName, "#Blob") == 0) {
				MetaData_LoadBlobs(pMetaData, pStream, streamSize);
			} else if (Gstrcasecmp((const char*)pStreamName, "#GUID") == 0) {
				MetaData_LoadGUIDs(pMetaData, pStream, streamSize);
			} else if (Gstrcasecmp((const char*)pStreamName, "#~") == 0) {
				pTableStream = pStream;
				tableStreamSize = streamSize;
			}
		}
		// Must load tables last
		if (pTableStream != NULL) {
			MetaData_LoadTables(pMetaData, pRet->pRVA, pTableStream, tableStreamSize);
		}
	}

	// Mark all generic definition types and methods as such
	//for (i=pMetaData->tables.numRows[MD_TABLE_GENERICPARAM]; i>0; i--) {
	//	tMD_GenericParam *pGenericParam;
	//	IDX_TABLE ownerIdx;

	//	pGenericParam = (tMD_GenericParam*)MetaData_GetTableRow
	//		(pMetaData, MAKE_TABLE_INDEX(MD_TABLE_GENERICPARAM, i));
	//	ownerIdx = pGenericParam->owner;
	//	switch (TABLE_ID(ownerIdx)) {
	//		case MD_TABLE_TYPEDEF:
	//			{
	//				tMD_TypeDef *pTypeDef = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, ownerIdx);
	//				pTypeDef->isGenericDefinition = 1;
	//			}
	//			break;
	//		case MD_TABLE_METHODDEF:
	//			{
	//				tMD_MethodDef *pMethodDef = (tMD_MethodDef*)MetaData_GetTableRow(pMetaData, ownerIdx);
	//				pMethodDef->isGenericDefinition = 1;
	//			}
	//			break;
	//		default:
	//			Crash("Wrong generic parameter owner: 0x%08x", ownerIdx);
	//	}
	//}

	// Mark all nested classes as such
	for (i=pMetaData->tables.numRows[MD_TABLE_NESTEDCLASS]; i>0; i--) {
		tMD_NestedClass *pNested;
		tMD_TypeDef *pParent, *pChild;

		pNested = (tMD_NestedClass*)MetaData_GetTableRow(pMetaData, MAKE_TABLE_INDEX(MD_TABLE_NESTEDCLASS, i));
		pParent = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, pNested->enclosingClass);
		pChild = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, pNested->nestedClass);
		pChild->pNestedIn = pParent;
	}

	return pRet;
}

function_space_specifier tCLIFile* CLIFile_Load(char *pFileName) {
	void *pRawFile;
	tCLIFile *pRet;
	tFilesLoaded *pNewFile;

	pRawFile = LoadFileFromDisk(pFileName);

	if (pRawFile == NULL) {
		Crash("Cannot load file: %s", pFileName);
	}

	pRet = LoadPEFile(pRawFile);
	pRet->pFileName = (char*)mallocForever((U32)Gstrlen(pFileName) + 1);
	Gstrcpy(pRet->pFileName, pFileName);

	// Record that we've loaded this file
	pNewFile = TMALLOCFOREVER(tFilesLoaded);
	pNewFile->pCLIFile = pRet;
	pNewFile->pNext = _bcl_->pFilesLoaded;
	_bcl_->pFilesLoaded = pNewFile;

	return pRet;
}

function_space_specifier I32 CLIFile_Execute(tCLIFile *pThis, int argc, char **argp) {
	//tThread *pThread;
	//HEAP_PTR args;
	//int i;

	//// Create a string array for the program arguments
	//// Don't include the argument that is the program name.
	//argc--;
	//argp++;
	//args = SystemArray_NewVector(types[TYPE_SYSTEM_ARRAY_STRING], argc);
	//Heap_MakeUndeletable(args);
	//for (i = 0; i < argc; i++) {
	//	HEAP_PTR arg = SystemString_FromCharPtrASCII(argp[i]);
	//	SystemArray_StoreElement(args, i, (PTR)&arg);
	//}

	//// Create the main application thread
	//pThread = Thread();
	//Thread_SetEntryPoint(pThread, pThis->pMetaData, pThis->entryPoint, (PTR)&args, sizeof(void*));

	//return Thread_Execute();
	return 0;
}

function_space_specifier void CLIFile_GetHeapRoots(tHeapRoots *pHeapRoots) {
	//tFilesLoaded *pFile;

	//pFile = pFilesLoaded;
	//while (pFile != NULL) {
	//	MetaData_GetHeapRoots(pHeapRoots, pFile->pCLIFile->pMetaData);
	//	pFile = pFile->pNext;
	//}
}
