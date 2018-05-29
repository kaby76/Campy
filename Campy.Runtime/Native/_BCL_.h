#pragma once

#include "Compat.h"
#include "Types.h"
#include <stdbool.h>

struct tJITCodeInfo_;
// struct tMD_TypeDef;
struct header_t;
struct tFilesLoaded_;
struct tMD_TypeDef_;
struct tHeapEntry_;
struct tLoadedLib_;
struct tArrayTypeDefs_;
struct tMD_MethodDef_;
struct tThread_;

struct _BCL_t {
	
	// Basics.
	void * global_memory_heap;
	void ** heap_list;
	int kernel_base_index;
	int count;
	int padding;
	int head_size;
	int pointer_count;

	// CLIFile.
	struct tFilesLoaded_ * pFilesLoaded;

	// Filesystem
	char** names;
	char** files;
	size_t* lengths;
	bool init;
	int initial_size;

	// Finalizer
	HEAP_PTR *ppToFinalize;
	int toFinalizeOfs;
	int toFinalizeCapacity;

	// Gstring
	char * ___strtok;

	// Heap
	struct tHeapEntry_ *pHeapTreeRoot;
	struct tHeapEntry_ *nil;
	U32 trackHeapSize;
	U32 heapSizeMax;
	U32 numNodes;
	U32 numCollections;

	// JIT_Execute
	struct tJITCodeInfo_ * jitCodeInfo;
	struct tJITCodeInfo_ * jitCodeGoNext;

	// MetaData
	unsigned int * tableRowSize;

	// Pinvoke
	struct tLoadedLib_ *pLoadedLibs;

	// Sys
	U32 logLevel;
	char * methodName;
	U32 mallocForeverSize;

	// Type
	struct tArrayTypeDefs_ *pArrays;
	U8 genericArrayMethodsInited;
	struct tMD_MethodDef_ ** ppGenericArrayMethods;
	struct tMD_TypeDef_ **types;
	U32 numInitTypes;

	// System.Console
	U32 nextKeybC;

	// Thread
	struct tThread_ *pAllThreads;
	struct tThread_ *pCurrentThread;

	// Type
	int CorLibDone;

	// Options -- bitmap of whether to print debugging information.
	U64 options;
};

#define BCL_DEBUG_PRINT_EVERYTHING 0x1
#define BCL_DEBUG_CHECK_HEAPS      0x2
#define BCL_DEBUG_INTERACTIVE      0x4
#define BCL_DEBUG_FUNCTION_ENTRY   0x8

extern gpu_space_specifier struct _BCL_t * _bcl_;
