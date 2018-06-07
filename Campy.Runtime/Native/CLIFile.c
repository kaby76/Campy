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
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#if defined(__GNUC__) && !CUDA
#include <experimental/filesystem>
#include <string>
#endif
#if defined(_MSC_VER) && !CUDA
#include <filesystem>
#include <string>
#include <direct.h>
#endif

// Is this exe/dll file for the .NET virtual machine?
#define IMAGE_FILE_MACHINE_I386 0x14c
#define IMAGE_FILE_MACHINE_AMD64 0x8664

typedef struct tFilesLoaded_ tFilesLoaded;
struct tFilesLoaded_ {
	tCLIFile *pCLIFile;
	tFilesLoaded *pNext;
};

// Keep track of all the files currently loaded
//function_space_specifier static tFilesLoaded *pFilesLoaded = NULL;
function_space_specifier tMetaData* CLIFile_GetMetaDataForAssembly(tMD_AssemblyRef * ar)
{
	return CLIFile_GetMetaDataForAssemblyAux(ar->name, ar->public_key_str, ar->majorVersion, ar->minorVersion);
}

function_space_specifier tMetaData* CLIFile_GetMetaDataForAssemblyAux(char * fileName, char* publickey, U16 majv, U16 minv)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("CLIFile_GetMetaDataForAssembly\n");

	// As there is no file names for module names, construct one.
	tFilesLoaded *pFiles;
	char * pAssemblyName;
	const int max_size = 250;
	char assemblyName[max_size];
	char fName[250];
	Gstrcpy(assemblyName, fileName);
	Gstrcpy(fName, fileName);

	// First, erase suffix to get assembly name.
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

	// Look in loaded files first
	pFiles = _bcl_->pFilesLoaded;
	while (pFiles != NULL)
	{
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
#if defined(__GNUC__) && !CUDA
		int added = 0;
		// Try current directory first.
		char * cwddir = getcwd(fName, max_size);
		if (cwddir != NULL)
		{
			std::experimental::filesystem::path p(fName);
			std::experimental::filesystem::directory_iterator start(p);
			std::experimental::filesystem::directory_iterator end;
			for (; start != end; ++start)
			{
				std::experimental::filesystem::path tpath = start->path();
				std::string path_string = tpath.u8string();
				std::string tail = path_string.substr(path_string.length() - 4);
				printf("%s\n", tail.c_str());
				printf("%s\n", path_string.c_str());
				strncpy(fName, assemblyName, max_size);
				strncat(fName, ".dll", max_size);
				std::size_t f1 = path_string.find(fName, 0);
				if (f1 != std::string::npos)
				{
					std::experimental::filesystem::path fn = tpath.filename();
					std::string fn2 = fn.u8string();
					const char * file_name = fn2.c_str();
					FILE * file = fopen(path_string.c_str(), "rb");
					fseek(file, 0, SEEK_END);
					long file_len = ftell(file);
					fseek(file, 0, SEEK_SET);
					char * buffer = (char *)Gmalloc(file_len + 1);
					if (!buffer)
					{
						fprintf(stderr, "Memory error!");
						fclose(file);
					}
					fread(buffer, file_len, 1, file);
					fclose(file);
					Gfs_add_file_no_malloc(file_name, buffer, file_len);
					added = 1;
					break;
				}
			}
		}
#endif
#if (defined(_MSC_VER) && !CUDA)
		int added = 0;
		// Try current directory first.
		char * cwddir = _getcwd(fName, max_size);
		if (cwddir != NULL)
		{
			std::experimental::filesystem::path p(fName);
			std::experimental::filesystem::directory_iterator start(p);
			std::experimental::filesystem::directory_iterator end;
			for (; start != end; ++start)
			{
				std::experimental::filesystem::path tpath = start->path();
				std::string path_string = tpath.u8string();
				std::string tail = path_string.substr(path_string.length() - 4);
				printf("%s\n", tail.c_str());
				printf("%s\n", path_string.c_str());
				strncpy(fName, assemblyName, max_size);
				strncat(fName, ".dll", max_size);
				std::size_t f1 = path_string.find(fName, 0);
				if (f1 != std::string::npos)
				{
					std::experimental::filesystem::path fn = tpath.filename();
					std::string fn2 = fn.u8string();
					const char * file_name = fn2.c_str();
					FILE * file = fopen(path_string.c_str(), "rb");
					fseek(file, 0, SEEK_END);
					long file_len = ftell(file);
					fseek(file, 0, SEEK_SET);
					char * buffer = (char *)Gmalloc(file_len + 1);
					if (!buffer)
					{
						fprintf(stderr, "Memory error!");
						fclose(file);
					}
					fread(buffer, file_len, 1, file);
					fclose(file);
					Gfs_add_file_no_malloc(file_name, buffer, file_len);
					added = 1;
					break;
				}
			}
		}
#endif
#if (defined(_MSC_VER) && !CUDA)
		// Functionality of assembly resolution seems it should be here.
		// But, this is a problem with GPU BCL because we don't have a
		// GAC for file system. What a mess.
		// For Windows, GAC is located in %windir%\Microsoft.NET\assembly
		// Look under preferentially GAC_64, then GAC_MSIL. Do not consider GAC_32 -- not supported.
		// The structure of all these files is GAC_*/<assembly-name>/<version>_<publickey>/<assembly-name>.dll
		char * windir = getenv("windir");
		if (added == 0 && windir != NULL)
		{
			strncpy(fName, windir, max_size);
			strncat(fName, "\\Microsoft.NET\\assembly\\GAC_MSIL\\", max_size);
			strncat(fName, assemblyName, max_size);
			strncat(fName, "\\", max_size);
			std::experimental::filesystem::path p(fName);
			std::experimental::filesystem::recursive_directory_iterator start(p);
			std::experimental::filesystem::recursive_directory_iterator end;
			for (; start != end; ++start) {
				std::experimental::filesystem::path tpath = start->path();
				std::string path_string = tpath.u8string();
				std::string tail = path_string.substr(path_string.length() - 4);
				printf("%s\n", tail.c_str());
				printf("%s\n", path_string.c_str());
				std::size_t f1 = path_string.find(publickey, 0);
				std::size_t f2 = tail.find(".dll", 0);
				if (f1 != std::string::npos && f2 != std::string::npos)
				{
					std::experimental::filesystem::path fn = tpath.filename();
					std::string fn2 = fn.u8string();
					const char * file_name = fn2.c_str();
					FILE * file = fopen(path_string.c_str(), "rb");
					fseek(file, 0, SEEK_END);
					long file_len = ftell(file);
					fseek(file, 0, SEEK_SET);
					char * buffer = (char *)Gmalloc(file_len + 1);
					if (!buffer)
					{
						fprintf(stderr, "Memory error!");
						fclose(file);
					}
					fread(buffer, file_len, 1, file);
					fclose(file);
					Gfs_add_file_no_malloc(file_name, buffer, file_len);
					added = 1;
					break;
				}
			}
		}
		// Net Core versioning does not work like Net Framework--wonderful designing by profession programmers!
		// Try to find appropriate Net Core file.
		if (added == 0)
		{
			strncpy(fName, "C:\\Program Files\\dotnet\\shared\\Microsoft.NETCore.App\\", max_size);
			std::experimental::filesystem::path p(fName);
			std::experimental::filesystem::recursive_directory_iterator start(p);
			std::experimental::filesystem::recursive_directory_iterator end;
			for (; start != end; ++start) {
				std::experimental::filesystem::path tpath = start->path();
				std::string path_string = tpath.u8string();
				std::string tail = path_string.substr(path_string.length() - 4);
				printf("%s\n", tail.c_str());
				printf("%s\n", path_string.c_str());
				printf("%s\n", path_string.c_str());
				strncpy(fName, assemblyName, max_size);
				strncat(fName, ".dll", max_size);
				std::size_t f1 = path_string.find(fName, 0);
				std::size_t f2 = path_string.find("\\2.0", 0);
				if (f1 != std::string::npos && f2 != std::string::npos)
				{
					std::experimental::filesystem::path fn = tpath.filename();
					std::string fn2 = fn.u8string();
					const char * file_name = fn2.c_str();
					FILE * file = fopen(path_string.c_str(), "rb");
					fseek(file, 0, SEEK_END);
					long file_len = ftell(file);
					fseek(file, 0, SEEK_SET);
					char * buffer = (char *)Gmalloc(file_len + 1);
					if (!buffer)
					{
						fprintf(stderr, "Memory error!");
						fclose(file);
					}
					fread(buffer, file_len, 1, file);
					fclose(file);
					Gfs_add_file_no_malloc(file_name, buffer, file_len);
					added = 1;
					break;
				}
			}
		}


#endif

		tCLIFile *pCLIFile;
		Gstrcpy(fName, pAssemblyName);
		Gstrcat(fName, ".dll");
		pCLIFile = CLIFile_Load(fName);
		if (pCLIFile != NULL)
			return pCLIFile->pMetaData;

		Gstrcpy(fName, pAssemblyName);
		Gstrcat(fName, ".exe");
		pCLIFile = CLIFile_Load(fName);
		if (pCLIFile != NULL)
			return pCLIFile->pMetaData;

		Crash("Cannot load required assembly %s either dll or exe.", pAssemblyName);
		return NULL;
	}
}

// function_space_specifier unsigned char Gdata[];

function_space_specifier static void* LoadFileFromDisk(char *pFileName)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("LoadFileFromDisk\n");
	char *pData = NULL;
	int handle;
	Gfs_open_file(pFileName, &handle);
	Gfs_read(handle, &pData);
#ifndef CUDA
	if (handle == -1)
	{
		// For host OS, try this if it's not in the GFS.
		int f = open(pFileName, O_RDONLY | O_BINARY);
		if (f >= 0)
		{
			int len;
			len = lseek(f, 0, SEEK_END);
			lseek(f, 0, SEEK_SET);
			// TODO: Change to use mmap() or windows equivilent
			pData = (char *)Gmalloc(len);
			if (pData != NULL) {
				int r = read(f, pData, len);
				if (r != len) {
					Gfree(pData);
					pData = NULL;
				}
			}
			close(f);
			int re;
			Gfs_add_file(pFileName, pData, len, &re);
		}
	}
#endif

	return pData;
}


function_space_specifier static tCLIFile* LoadPEFile(char * pFileName, void *pData)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("LoadPEFile\n");
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
	pMetaData->file_name = Gstrdup(pFileName);

	struct BCL_IMAGE_SECTION_HEADER * psh;

	// Cast dos header pointer to the above BCL_IMAGE_DOS_HEADER struct.
	struct BCL_IMAGE_DOS_HEADER * p_dos_header = (struct BCL_IMAGE_DOS_HEADER *)pMSDOSHeader;

	lfanew = p_dos_header->e_lfanew;

	pPEHeader = pMSDOSHeader + lfanew + 4;
	struct BCL_IMAGE_NT_HEADERS * pnt = (struct BCL_IMAGE_NT_HEADERS*)pPEHeader;
	struct BCL_IMAGE_FILE_HEADER * ph = (struct BCL_IMAGE_FILE_HEADER *)pPEHeader;

	pPEOptionalHeader = pPEHeader + 20;
	pPESectionHeaders = pPEOptionalHeader + 224;

	machine = ph->Machine;

	if (!(machine == IMAGE_FILE_MACHINE_I386 || machine == IMAGE_FILE_MACHINE_AMD64))
	{
		Gprintf("Not IMAGE_FILE_MACHINE_I386 or .\n");
		return NULL;
	}

	numSections = ph->NumberOfSections;
	imageBase = GetU32(&(pPEOptionalHeader[28]));
	fileAlignment = GetU32(&(pPEOptionalHeader[36]));

#define _htol16(x) (x)

	ph->NumberOfSections = _htol16(ph->NumberOfSections);
	ph->SizeOfOptionalHeader = _htol16(ph->SizeOfOptionalHeader);
	off_t base = 0;
	int cnt = 0;
	unsigned int ncliHeaderRVA;
	unsigned int ncliHeaderSize;
	if (ph->SizeOfOptionalHeader >= sizeof(*ph))
	{
		struct BCL_IMAGE_OPTIONAL_HEADER32 * pih = (struct BCL_IMAGE_OPTIONAL_HEADER32 *)pPEOptionalHeader;
		if (pih->Magic == IMAGE_NT_OPTIONAL_HDR32_MAGIC
			&& ph->SizeOfOptionalHeader >= sizeof(*pih))
		{
			/* PE32 executable */
			struct BCL_IMAGE_OPTIONAL_HEADER32 *pih32 = (struct BCL_IMAGE_OPTIONAL_HEADER32 *)(pih);
#define _htol32(x) (x)
			pih32->ImageBase = _htol32(pih32->ImageBase);
			pih32->NumberOfRvaAndSizes = _htol32(pih32->NumberOfRvaAndSizes);
			base = pih32->ImageBase;
			cliHeaderRVA = pih32->DataDirectory[0xe].VirtualAddress;
			cliHeaderSize = pih32->DataDirectory[0xe].Size;
		}
		else if (pih->Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC
			&& ph->SizeOfOptionalHeader >= sizeof(*pih))
		{
			/* PE32+ executable */
			struct BCL_IMAGE_OPTIONAL_HEADER64 * pih32p = (struct BCL_IMAGE_OPTIONAL_HEADER64 *)(pih);
#define _htol64(x) (x)
			cliHeaderRVA = pih32p->DataDirectory[0xe].VirtualAddress;
			cliHeaderSize = pih32p->DataDirectory[0xe].Size;
		}
	}

	/* read and record each section */
	char _pe_msdos_signature[2];
	memcpy(_pe_msdos_signature, "MZ", 2);
	char _pe_header_signature[4];
	memcpy(_pe_header_signature, "PE\0\0", 4);
	off_t offset = p_dos_header->e_lfanew;
	offset += sizeof(_pe_header_signature);
	offset += sizeof(*ph);
	offset += ph->SizeOfOptionalHeader;

	for (i=0; i<numSections; i++) {
		unsigned char *pSection = pMSDOSHeader + offset;
		RVA_Create(pRet->pRVA, pData, pSection);
		offset += sizeof(*psh);
	}

	pCLIHeader = (unsigned char *)RVA_FindData(pRet->pRVA, cliHeaderRVA);
	if (pCLIHeader == NULL)
		return NULL;
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
	for (i=pMetaData->tables.numRows[MD_TABLE_GENERICPARAM]; i>0; i--) {
		tMD_GenericParam *pGenericParam;
		IDX_TABLE ownerIdx;

		pGenericParam = (tMD_GenericParam*)MetaData_GetTableRow
			(pMetaData, MAKE_TABLE_INDEX(MD_TABLE_GENERICPARAM, i));
		ownerIdx = pGenericParam->owner;
		switch (TABLE_ID(ownerIdx)) {
			case MD_TABLE_TYPEDEF:
				{
					tMD_TypeDef *pTypeDef = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, ownerIdx);
					pTypeDef->isGenericDefinition = 1;
				}
				break;
			case MD_TABLE_METHODDEF:
				{
					tMD_MethodDef *pMethodDef = (tMD_MethodDef*)MetaData_GetTableRow(pMetaData, ownerIdx);
					pMethodDef->isGenericDefinition = 1;
				}
				break;
			default:
				Crash("Wrong generic parameter owner: 0x%08x", ownerIdx);
		}
	}

	// Mark all nested classes as such
	for (i=pMetaData->tables.numRows[MD_TABLE_NESTEDCLASS]; i>0; i--) {
		tMD_NestedClass *pNested;
		tMD_TypeDef *pParent, *pChild;

		long long int vvv = MAKE_TABLE_INDEX(MD_TABLE_NESTEDCLASS, i);

		// There seems to be some bugs here....
		// For now, if there's a problem, just warn.
		pNested = (tMD_NestedClass*)MetaData_GetTableRow(pMetaData, MAKE_TABLE_INDEX(MD_TABLE_NESTEDCLASS, i));
		pParent = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, pNested->enclosingClass);
		pChild = (tMD_TypeDef*)MetaData_GetTableRow(pMetaData, pNested->nestedClass);
		if (pChild == NULL)
		{
			Gprintf("Warning--meta reading is messed up.\n");
		}
		else
		{
			pChild->pNestedIn = pParent;
		}
	}

	return pRet;
}

function_space_specifier tCLIFile* CLIFile_Load(char *pFileName)
{
	if (_bcl_ && _bcl_->options & BCL_DEBUG_FUNCTION_ENTRY)
		Gprintf("CLIFile_Load\n");
	
	void *pRawFile;
	tCLIFile *pRet;
	tFilesLoaded *pNewFile;

	pRawFile = LoadFileFromDisk(pFileName);

	if (pRawFile == NULL)
	{
		Gprintf("Warning: assembly %s cannot be found.\n", pFileName);
		return NULL;
	}

	pRet = LoadPEFile(pFileName, pRawFile);
	if (pRet == NULL)
	{
		Gprintf("Warning: assembly %s does not appear to have metadata.\n", pFileName);
		return NULL;
	}
	pRet->pFileName = (char*)mallocForever((U32)Gstrlen(pFileName) + 1);
	Gstrcpy(pRet->pFileName, pFileName);

	// Record that we've loaded this file
	pNewFile = TMALLOCFOREVER(tFilesLoaded);
	pNewFile->pCLIFile = pRet;
	pNewFile->pNext = _bcl_->pFilesLoaded;
	_bcl_->pFilesLoaded = pNewFile;

	return pRet;
}

function_space_specifier I32 CLIFile_Execute(tCLIFile *pThis, int argc, char **argp)
{
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

function_space_specifier void CLIFile_GetHeapRoots(tHeapRoots *pHeapRoots)
{
	//tFilesLoaded *pFile;

	//pFile = pFilesLoaded;
	//while (pFile != NULL) {
	//	MetaData_GetHeapRoots(pHeapRoots, pFile->pCLIFile->pMetaData);
	//	pFile = pFile->pNext;
	//}
}
