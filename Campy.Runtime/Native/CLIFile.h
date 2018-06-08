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
typedef struct tCLIFile_ tCLIFile;

// static methods
function_space_specifier tMetaData* CLIFile_GetMetaDataForAssembly(tMD_AssemblyRef * ar);
function_space_specifier tMetaData* CLIFile_GetMetaDataForAssemblyAux(char * fileName, char* publickey, U16 majv, U16 minv);
function_space_specifier void CLIFile_GetHeapRoots(tHeapRoots *pHeapRoots);

// instance methods
function_space_specifier tCLIFile* CLIFile_Load(char *pFileName);
function_space_specifier I32 CLIFile_Execute(tCLIFile *pThis, int argc, char **argp);



#pragma pack(1)
struct BCL_IMAGE_DOS_HEADER {
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
};

#define IMAGE_DOS_SIGNATURE		0x5A4D     /* MZ   */
#define IMAGE_NT_SIGNATURE		0x00004550 /* PE00 */

#define IMAGE_FILE_MACHINE_ARM		0x01c0
#define IMAGE_FILE_MACHINE_THUMB	0x01c2
#define IMAGE_FILE_MACHINE_ARMNT	0x01c4
#define IMAGE_FILE_MACHINE_AMD64	0x8664
#define IMAGE_FILE_MACHINE_ARM64	0xaa64
#define IMAGE_FILE_MACHINE_I386     0x014c
#define IMAGE_FILE_MACHINE_AMD64    0x8664
#define IMAGE_FILE_MACHINE_UNDOCUMENTED0xFD1D 0xfd1d


#define IMAGE_NT_OPTIONAL_HDR32_MAGIC	0x10b
#define IMAGE_NT_OPTIONAL_HDR64_MAGIC	0x20b
#define IMAGE_SUBSYSTEM_EFI_APPLICATION	10

struct BCL_IMAGE_FILE_HEADER {
	uint16_t Machine;
	uint16_t NumberOfSections;
	uint32_t TimeDateStamp;
	uint32_t PointerToSymbolTable;
	uint32_t NumberOfSymbols;
	uint16_t SizeOfOptionalHeader;
	uint16_t Characteristics;
};

struct BCL_IMAGE_DATA_DIRECTORY {
	uint32_t VirtualAddress;
	uint32_t Size;
};

#define IMAGE_NUMBEROF_DIRECTORY_ENTRIES 16

struct BCL_IMAGE_OPTIONAL_HEADER64 {
	uint16_t Magic; /* 0x20b */
	uint8_t  MajorLinkerVersion;
	uint8_t  MinorLinkerVersion;
	uint32_t SizeOfCode;
	uint32_t SizeOfInitializedData;
	uint32_t SizeOfUninitializedData;
	uint32_t AddressOfEntryPoint;
	uint32_t BaseOfCode;
	uint64_t ImageBase;
	uint32_t SectionAlignment;
	uint32_t FileAlignment;
	uint16_t MajorOperatingSystemVersion;
	uint16_t MinorOperatingSystemVersion;
	uint16_t MajorImageVersion;
	uint16_t MinorImageVersion;
	uint16_t MajorSubsystemVersion;
	uint16_t MinorSubsystemVersion;
	uint32_t Win32VersionValue;
	uint32_t SizeOfImage;
	uint32_t SizeOfHeaders;
	uint32_t CheckSum;
	uint16_t Subsystem;
	uint16_t DllCharacteristics;
	uint64_t SizeOfStackReserve;
	uint64_t SizeOfStackCommit;
	uint64_t SizeOfHeapReserve;
	uint64_t SizeOfHeapCommit;
	uint32_t LoaderFlags;
	uint32_t NumberOfRvaAndSizes;
	struct BCL_IMAGE_DATA_DIRECTORY DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
};

struct BCL_IMAGE_NT_HEADERS64 {
	uint32_t Signature;
	struct BCL_IMAGE_FILE_HEADER FileHeader;
	struct BCL_IMAGE_OPTIONAL_HEADER64 OptionalHeader;
};

struct BCL_IMAGE_OPTIONAL_HEADER32 {

	/* Standard fields */

	uint16_t Magic; /* 0x10b or 0x107 */     /* 0x00 */
	uint8_t  MajorLinkerVersion;
	uint8_t  MinorLinkerVersion;
	uint32_t SizeOfCode;
	uint32_t SizeOfInitializedData;
	uint32_t SizeOfUninitializedData;
	uint32_t AddressOfEntryPoint;            /* 0x10 */
	uint32_t BaseOfCode;
	uint32_t BaseOfData;

	/* NT additional fields */

	uint32_t ImageBase;
	uint32_t SectionAlignment;               /* 0x20 */
	uint32_t FileAlignment;
	uint16_t MajorOperatingSystemVersion;
	uint16_t MinorOperatingSystemVersion;
	uint16_t MajorImageVersion;
	uint16_t MinorImageVersion;
	uint16_t MajorSubsystemVersion;          /* 0x30 */
	uint16_t MinorSubsystemVersion;
	uint32_t Win32VersionValue;
	uint32_t SizeOfImage;
	uint32_t SizeOfHeaders;
	uint32_t CheckSum;                       /* 0x40 */
	uint16_t Subsystem;
	uint16_t DllCharacteristics;
	uint32_t SizeOfStackReserve;
	uint32_t SizeOfStackCommit;
	uint32_t SizeOfHeapReserve;              /* 0x50 */
	uint32_t SizeOfHeapCommit;
	uint32_t LoaderFlags;
	uint32_t NumberOfRvaAndSizes;
	struct BCL_IMAGE_DATA_DIRECTORY DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES]; /* 0x60 */
																		  /* 0xE0 */
};

struct BCL_IMAGE_NT_HEADERS {
	uint32_t Signature; /* "PE"\0\0 */       /* 0x00 */
	struct BCL_IMAGE_FILE_HEADER FileHeader;         /* 0x04 */
	struct BCL_IMAGE_OPTIONAL_HEADER32 OptionalHeader;       /* 0x18 */
};

#define IMAGE_SIZEOF_SHORT_NAME 8

struct BCL_IMAGE_SECTION_HEADER {
	uint8_t	Name[IMAGE_SIZEOF_SHORT_NAME];
	union {
		uint32_t PhysicalAddress;
		uint32_t VirtualSize;
	} Misc;
	uint32_t VirtualAddress;
	uint32_t SizeOfRawData;
	uint32_t PointerToRawData;
	uint32_t PointerToRelocations;
	uint32_t PointerToLinenumbers;
	uint16_t NumberOfRelocations;
	uint16_t NumberOfLinenumbers;
	uint32_t Characteristics;
};

#define IMAGE_DIRECTORY_ENTRY_BASERELOC         5

struct BCL_IMAGE_BASE_RELOCATION
{
	uint32_t VirtualAddress;
	uint32_t SizeOfBlock;
	/* WORD TypeOffset[1]; */
};

struct BCL_IMAGE_RELOCATION
{
	union {
		uint32_t VirtualAddress;
		uint32_t RelocCount;
	} DUMMYUNIONNAME;
	uint32_t SymbolTableIndex;
	uint16_t Type;
};

#define IMAGE_SIZEOF_RELOCATION 10

/* generic relocation types */
#define IMAGE_REL_BASED_ABSOLUTE                0
#define IMAGE_REL_BASED_HIGH                    1
#define IMAGE_REL_BASED_LOW                     2
#define IMAGE_REL_BASED_HIGHLOW                 3
#define IMAGE_REL_BASED_HIGHADJ                 4
#define IMAGE_REL_BASED_MIPS_JMPADDR            5
#define IMAGE_REL_BASED_ARM_MOV32A              5 /* yes, 5 too */
#define IMAGE_REL_BASED_ARM_MOV32               5 /* yes, 5 too */
#define IMAGE_REL_BASED_SECTION                 6
#define IMAGE_REL_BASED_REL                     7
#define IMAGE_REL_BASED_ARM_MOV32T              7 /* yes, 7 too */
#define IMAGE_REL_BASED_THUMB_MOV32             7 /* yes, 7 too */
#define IMAGE_REL_BASED_MIPS_JMPADDR16          9
#define IMAGE_REL_BASED_IA64_IMM64              9 /* yes, 9 too */
#define IMAGE_REL_BASED_DIR64                   10
#define IMAGE_REL_BASED_HIGH3ADJ                11

/* ARM relocation types */
#define IMAGE_REL_ARM_ABSOLUTE          0x0000
#define IMAGE_REL_ARM_ADDR              0x0001
#define IMAGE_REL_ARM_ADDR32NB          0x0002
#define IMAGE_REL_ARM_BRANCH24          0x0003
#define IMAGE_REL_ARM_BRANCH11          0x0004
#define IMAGE_REL_ARM_TOKEN             0x0005
#define IMAGE_REL_ARM_GPREL12           0x0006
#define IMAGE_REL_ARM_GPREL7            0x0007
#define IMAGE_REL_ARM_BLX24             0x0008
#define IMAGE_REL_ARM_BLX11             0x0009
#define IMAGE_REL_ARM_SECTION           0x000E
#define IMAGE_REL_ARM_SECREL            0x000F
#define IMAGE_REL_ARM_MOV32A            0x0010
#define IMAGE_REL_ARM_MOV32T            0x0011
#define IMAGE_REL_ARM_BRANCH20T         0x0012
#define IMAGE_REL_ARM_BRANCH24T         0x0014
#define IMAGE_REL_ARM_BLX23T            0x0015

/* ARM64 relocation types */
#define IMAGE_REL_ARM64_ABSOLUTE        0x0000
#define IMAGE_REL_ARM64_ADDR32          0x0001
#define IMAGE_REL_ARM64_ADDR32NB        0x0002
#define IMAGE_REL_ARM64_BRANCH26        0x0003
#define IMAGE_REL_ARM64_PAGEBASE_REL21  0x0004
#define IMAGE_REL_ARM64_REL21           0x0005
#define IMAGE_REL_ARM64_PAGEOFFSET_12A  0x0006
#define IMAGE_REL_ARM64_PAGEOFFSET_12L  0x0007
#define IMAGE_REL_ARM64_SECREL          0x0008
#define IMAGE_REL_ARM64_SECREL_LOW12A   0x0009
#define IMAGE_REL_ARM64_SECREL_HIGH12A  0x000A
#define IMAGE_REL_ARM64_SECREL_LOW12L   0x000B
#define IMAGE_REL_ARM64_TOKEN           0x000C
#define IMAGE_REL_ARM64_SECTION         0x000D
#define IMAGE_REL_ARM64_ADDR64          0x000E

/* AMD64 relocation types */
#define IMAGE_REL_AMD64_ABSOLUTE        0x0000
#define IMAGE_REL_AMD64_ADDR64          0x0001
#define IMAGE_REL_AMD64_ADDR32          0x0002
#define IMAGE_REL_AMD64_ADDR32NB        0x0003
#define IMAGE_REL_AMD64_REL32           0x0004
#define IMAGE_REL_AMD64_REL32_1         0x0005
#define IMAGE_REL_AMD64_REL32_2         0x0006
#define IMAGE_REL_AMD64_REL32_3         0x0007
#define IMAGE_REL_AMD64_REL32_4         0x0008
#define IMAGE_REL_AMD64_REL32_5         0x0009
#define IMAGE_REL_AMD64_SECTION         0x000A
#define IMAGE_REL_AMD64_SECREL          0x000B
#define IMAGE_REL_AMD64_SECREL7         0x000C
#define IMAGE_REL_AMD64_TOKEN           0x000D
#define IMAGE_REL_AMD64_SREL32          0x000E
#define IMAGE_REL_AMD64_PAIR            0x000F
#define IMAGE_REL_AMD64_SSPAN32         0x0010
#pragma pack()


#endif
