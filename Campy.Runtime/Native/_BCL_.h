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
	struct header_t * head;
	int kernel_base_index;

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
};

extern gpu_space_specifier struct _BCL_t * _bcl_;
global_space_specifier void Initialize_BCL_Globals(void * g, size_t size, size_t first_overhead, int count, struct _BCL_t ** pbcl);
