#pragma once

#include "Compat.h"
#include "Types.h"

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
	void * global_memory_heap;
	struct header_t * head;
	tFilesLoaded_ * pFilesLoaded;
	unsigned char * Gdata;

	// Filesystem
	char** names;
	char** files;
	size_t* lengths;
	boolean init;
	int initial_size;

	// Finalizer
	HEAP_PTR *ppToFinalize;
	int toFinalizeOfs, toFinalizeCapacity;

	// Gstring
	char * ___strtok;

	// Heap
	tHeapEntry_ *pHeapTreeRoot;
	tHeapEntry_ *nil;
	U32 trackHeapSize;
	U32 heapSizeMax;
	U32 numNodes;
	U32 numCollections;

	// JIT_Execute
	tJITCodeInfo_ * jitCodeInfo;
	tJITCodeInfo_ * jitCodeGoNext;

	// MetaData
	unsigned int * tableRowSize;

	// Pinvoke
	tLoadedLib_ *pLoadedLibs;

	// Sys
	U32 logLevel;
	char * methodName;
	U32 mallocForeverSize;

	// Type
	tArrayTypeDefs_ *pArrays;
	U8 genericArrayMethodsInited = 0;
	tMD_MethodDef_ ** ppGenericArrayMethods;
	tMD_TypeDef_ **types;
	U32 numInitTypes;

	// System.Console
	U32 nextKeybC;

	// Thread
	tThread_ *pAllThreads;
	tThread_ *pCurrentThread;

	// Type
	int CorLibDone;
};

extern __device__ _BCL_t * _bcl_;
