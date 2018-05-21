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

#if !defined (__CLIFILE_H)
#define __CLIFILE_H

#include "RVA.h"
#include "Types.h"
#include "MetaData.h"

typedef struct tCLIFile_ tCLIFile;
struct tCLIFile_ {
	// The filename
	char *pFileName;
	// The RVA sections of this file
	tRVA *pRVA;
	// NULL-terminated UTF8 string of file version
	unsigned char *pVersion;
	// The entry point token if this is executable, 0 if it isn't
	IDX_TABLE entryPoint;

	tMetaData *pMetaData;
};

// static methods
function_space_specifier tMetaData* CLIFile_GetMetaDataForAssembly(char *pAssemblyName);
function_space_specifier void CLIFile_GetHeapRoots(tHeapRoots *pHeapRoots);

// instance methods
function_space_specifier tCLIFile* CLIFile_Load(char *pFileName);
function_space_specifier I32 CLIFile_Execute(tCLIFile *pThis, int argc, char **argp);



#pragma pack(1)
typedef struct dos_header {
	uint16_t e_magic;	/* 00: MZ Header signature */
	uint16_t e_cblp;	/* 02: Bytes on last page of file */
	uint16_t e_cp;		/* 04: Pages in file */
	uint16_t e_crlc;	/* 06: Relocations */
	uint16_t e_cparhdr;	/* 08: Size of header in paragraphs */
	uint16_t e_minalloc;	/* 0a: Minimum extra paragraphs needed */
	uint16_t e_maxalloc;	/* 0c: Maximum extra paragraphs needed */
	uint16_t e_ss;		/* 0e: Initial (relative) SS value */
	uint16_t e_sp;		/* 10: Initial SP value */
	uint16_t e_csum;	/* 12: Checksum */
	uint16_t e_ip;		/* 14: Initial IP value */
	uint16_t e_cs;		/* 16: Initial (relative) CS value */
	uint16_t e_lfarlc;	/* 18: File address of relocation table */
	uint16_t e_ovno;	/* 1a: Overlay number */
	uint16_t e_res[4];	/* 1c: Reserved words */
	uint16_t e_oemid;	/* 24: OEM identifier (for e_oeminfo) */
	uint16_t e_oeminfo;	/* 26: OEM information; e_oemid specific */
	uint16_t e_res2[10];	/* 28: Reserved words */
	uint32_t e_lfanew;	/* 3c: Offset to extended header */
} BCL_IMAGE_DOS_HEADER;

#define IMAGE_DOS_SIGNATURE		0x5A4D     /* MZ   */
#define IMAGE_NT_SIGNATURE		0x00004550 /* PE00 */

#define IMAGE_FILE_MACHINE_ARM		0x01c0
#define IMAGE_FILE_MACHINE_THUMB	0x01c2
#define IMAGE_FILE_MACHINE_ARMNT	0x01c4
#define IMAGE_FILE_MACHINE_AMD64	0x8664
#define IMAGE_FILE_MACHINE_ARM64	0xaa64
#define IMAGE_NT_OPTIONAL_HDR32_MAGIC	0x10b
#define IMAGE_NT_OPTIONAL_HDR64_MAGIC	0x20b
#define IMAGE_SUBSYSTEM_EFI_APPLICATION	10

struct pe_header
{
	uint16_t machine;
	uint16_t section_cnt;
	uint32_t timestamp;
	uint32_t symbol_offset;
	uint32_t symbol_cnt;
	uint16_t opthdr_size;
	uint16_t flags;
};

struct pe_image_header
{
	uint16_t signature;
	uint8_t major;
	uint8_t minor;
	uint32_t code_size;
	uint32_t code_initialized;
	uint32_t code_uninitialized;
	uint32_t entrypoint;
	uint32_t code_base;
};

struct pe_image_header_data
{
	uint32_t vaddr;
	uint32_t size;
};

#define IMAGE_NUMBEROF_DIRECTORY_ENTRIES 16

struct pe_image_header_pe32
{
	uint32_t data_base;
	uint32_t image_base;
	uint32_t section_alignment;
	uint32_t file_alignment;
	uint16_t os_major;
	uint16_t os_minor;
	uint16_t image_major;
	uint16_t image_minor;
	uint16_t subsys_major;
	uint16_t subsys_minor;
	uint32_t win32_version;
	uint32_t image_size;
	uint32_t headers_size;
	uint32_t checksum;
	uint16_t subsys;
	uint16_t dll_flags;
	uint32_t stack_reserved;
	uint32_t stack_commit;
	uint32_t heap_reserved;
	uint32_t heap_commit;
	uint32_t loader_flags;
	uint32_t rvasizes_cnt;
	//pe_image_header_data DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
};

#define IMAGE_NUMBEROF_DIRECTORY_ENTRIES 16

struct pe_image_header_pe32_plus // magic # = 0x20b.
{
	uint64_t image_base;
	uint32_t section_alignment;
	uint32_t file_alignment;
	uint16_t os_major;
	uint16_t os_minor;
	uint16_t image_major;
	uint16_t image_minor;
	uint16_t subsys_major;
	uint16_t subsys_minor;
	uint32_t win32_version;
	uint32_t image_size;
	uint32_t headers_size;
	uint32_t checksum;
	uint16_t subsys;
	uint16_t dll_flags;
	uint32_t stack_reserved;
	uint64_t stack_commit;
	uint64_t heap_reserved;
	uint64_t heap_commit;
	uint32_t loader_flags;
	uint32_t rvasizes_cnt;
	//pe_image_header_data DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
};

struct pe_msdos
{
	char signature[2];
	char _padding[0x3a];
	uint16_t offset;
};


struct pe_symbol
{
	union
	{
		struct
		{
			char name[8];
		} _short;
		struct
		{
			uint32_t zero;
			uint32_t offset;
		} _long;
	} name;
	uint32_t value;
	uint16_t section;
	uint16_t type;
	uint8_t storage_class;
	uint8_t aux_cnt;
};

struct pe_section_header
{
	char name[8];
	union
	{
		uint32_t paddr;
		uint32_t vsize;
	} misc;
	uint32_t vaddr;
	uint32_t raw_size;
	uint32_t raw_offset;
	uint32_t raw_reloc;
	uint32_t lines_offset;
	uint16_t reloc_cnt;
	uint16_t lines_cnt;
	uint32_t flags;
};

#pragma pack()

/* constants */
/* program header machine types */
#define PE_IMAGE_FILE_MACHINE_AMD64	0x8664
#define PE_IMAGE_FILE_MACHINE_ARM	0x1c00
#define PE_IMAGE_FILE_MACHINE_I386	0x014c
#define PE_IMAGE_FILE_MACHINE_UNKNOWN	0x0000

/* program image header signatures */
#define PE_IMAGE_HEADER_ROM		0x0107
#define PE_IMAGE_HEADER_PE32		0x010b
#define PE_IMAGE_HEADER_PE32_PLUS	0x020b

/* section header flags */
#define PE_IMAGE_SCN_CNT_CODE		0x0000002


#endif
