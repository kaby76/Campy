
#include "_bcl_.h"
#include "Sys.h"
#include "MetaData.h"
#include "JIT.h"
#include "Type.h"
#include "Finalizer.h"
#include "System.Net.Sockets.Socket.h"
#include "Gprintf.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

gpu_space_specifier struct _BCL_t * _bcl_;
function_space_specifier void Initialize_BCL0(void * g, size_t size, int count);

global_space_specifier void Initialize_BCL_Globals(void * g, size_t size, int count, struct _BCL_t ** pbcl)
{
	// basics/memory allocation.
	struct _BCL_t * bcl = (struct _BCL_t*)g;
	*pbcl = bcl;
	_bcl_ = bcl;
	memset(bcl, 0, sizeof(struct _BCL_t));
	_bcl_->global_memory_heap = NULL;
	_bcl_->head = NULL;

	// Init memory allocation.
	Initialize_BCL0(g, size, count);

	// CLIFile.
	bcl->pFilesLoaded = NULL;

	// Filesystem.
	bcl->names = NULL;
	bcl->files = NULL;
	bcl->lengths = NULL;
	bcl->init = 0;
	bcl->initial_size = 0;

	// Finalizer
	bcl->ppToFinalize = NULL;
	bcl->toFinalizeOfs = 0;
	bcl->toFinalizeCapacity = 0;

	// Gstring
	bcl->___strtok = NULL;

	// Heap
	bcl->pHeapTreeRoot = NULL;
	bcl->nil = NULL;
	bcl->trackHeapSize = 0;
	bcl->heapSizeMax = 0;
	bcl->numNodes = 0;
	bcl->numCollections = 0;

	// JIT_Execute
	bcl->jitCodeInfo = (struct tJITCodeInfo_ *) malloc(JIT_OPCODE_MAXNUM * sizeof(struct tJITCodeInfo_));
	memset(bcl->jitCodeInfo, 0, JIT_OPCODE_MAXNUM * sizeof(struct tJITCodeInfo_));
	bcl->jitCodeGoNext = (struct tJITCodeInfo_ *) malloc(1 * sizeof(struct tJITCodeInfo_));
	memset(bcl->jitCodeGoNext, 0, 1 * sizeof(struct tJITCodeInfo_));

	// MetaData
	bcl->tableRowSize = (unsigned int *) malloc(MAX_TABLES * sizeof(unsigned int));

	// Pinvoke
	bcl->pLoadedLibs = NULL;

	// Sys
	bcl->logLevel = 0;
	bcl->methodName = (char *)malloc(2048 * sizeof(char));
	bcl->mallocForeverSize = 0;

	// Type
	bcl->pArrays = NULL;
	bcl->genericArrayMethodsInited = 0;
	struct tMD_MethodDef_ ** ppGenericArrayMethods;
	bcl->ppGenericArrayMethods = (struct tMD_MethodDef_ **)malloc(GENERICARRAYMETHODS_NUM * sizeof(struct tMD_MethodDef_ *));
	bcl->types = NULL;
	bcl->numInitTypes = 0;

	// System.Console
	bcl->nextKeybC = 0xffffffff;

	// Thread
	bcl->pAllThreads = NULL;
	bcl->pCurrentThread = NULL;

	// Type
	bcl->CorLibDone = 0;
}

global_space_specifier void Set_BCL_Globals(struct _BCL_t * bcl)
{
	_bcl_ = bcl;
}

gpu_space_specifier void Get_BCL_Globals(struct _BCL_t ** bcl)
{
	*bcl = _bcl_;
}


function_space_specifier  void gpuexit(int _Code) {}

function_space_specifier size_t Gstrlen(
	char const* _Str
)
{
    size_t r = 0;
    for (size_t l = 0; _Str[l] != 0; r = ++l)
        ;
    return r;
}


// No good way to do mutex.
// https://devtalk.nvidia.com/default/topic/1014009/try-to-use-lock-and-unlock-in-cuda/?offset=1
// For now, per-thread heaps.
// Based on http://arjunsreedharan.org/post/148675821737/write-a-simple-memory-allocator


struct header_t {
	struct header_t *next;
	struct header_t *prev;
	size_t size;
	unsigned is_free;
};


function_space_specifier void Initialize_BCL0(void * g, size_t size, int count)
{
	// Initialize memory allocation / malloc. Nothing can be done until this is done.
	// Layout
	//
	//  ==================================
	//  0                         bcl
	//  0x1000                    headers
	//  0x1000+size_for_headers   ptr
	//  ==================================
	int size_for_bcl = 0x1000;
	_bcl_->head = (struct header_t*)(size_for_bcl + (unsigned char*)g);
	int size_for_headers = sizeof(struct header_t) * count;
	unsigned char * ptr = size_for_bcl + size_for_headers + (unsigned char *)_bcl_->head;
	int overhead_for_first = 16777216;
	long long s = (long long) ptr;
	long long e = s + size;

	int remainder = size - size_for_bcl - size_for_headers - overhead_for_first;
	int per_thread_remainder = remainder / count;

	per_thread_remainder = per_thread_remainder >> 3;
	per_thread_remainder = per_thread_remainder << 3;
	for (int c = 0; c < count; ++c)
	{
		struct header_t * start = &_bcl_->head[c];
		struct header_t * h = (struct header_t*)ptr;
		{
			long long hs = (long long)h;
			int diff = hs - s;
			if (diff >= size)
			{
				printf("out of bounds\n");
			}
		}
		int siz;
		if (c == 0)
			siz = overhead_for_first;
		else
			siz = per_thread_remainder;
		start->is_free = 0;
		start->next = h;
		start->prev = NULL;
		start->size = 0;
		int alloc_siz = siz - sizeof(struct header_t);
		h->next = NULL;
		h->prev = start;
		h->is_free = 1;
		h->size = alloc_siz;
		ptr += siz;
	}
	printf("Initialized memory allocation/malloc\n");

}

function_space_specifier struct header_t *get_free_block(size_t size)
{
#ifdef CUDA
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
#else
	int threadId = 0;
#endif

	struct header_t *curr = &_bcl_->head[threadId];
	while (curr) {
		if (curr->is_free && curr->size >= size)
			return curr;
		curr = curr->next;
	}
	return NULL;
}

function_space_specifier int roundUp(int numToRound, int multiple)
{
	return ((numToRound + multiple - 1) / multiple) * multiple;
}

function_space_specifier void * simple_malloc(size_t size)
{
	size_t total_size;
	void *block;
	struct header_t *header;
	if (!size)
		return NULL;
	size = roundUp(size, 8);
	header = get_free_block(size);
	if (header)
	{
		printf("header = %llx\n", header);
		// split block if big enough.
		if (header->size > (size + sizeof(struct header_t)))
		{
			int original_size = header->size;
			int skip = size + sizeof(struct header_t);
			unsigned char * ptr = ((unsigned char *)header) + skip;
			struct header_t * new_free = (struct header_t *)ptr;
			new_free->is_free = 1;
			new_free->size = original_size - skip;
			new_free->prev = header->prev;
			new_free->next = header->next;
			if (new_free->prev != NULL) new_free->prev->next = new_free;
			header->size = size;
		}
		header->is_free = 0;
		return (void*)(header + 1);
	}
	return NULL;
}


function_space_specifier void* __cdecl Grealloc(
	void*  _Block,
	size_t _Size
)
{
	void * result = simple_malloc(_Size);
	memcpy(result, _Block, _Size);
	Gfree(_Block);
    return result;
}


function_space_specifier void* Gmalloc(
	size_t _Size
)
{
	void * result = simple_malloc(_Size);
	return result;
}

function_space_specifier void Gfree(
	void*  _Block
	)
{
}


__global__
void Initialize_BCL1()
{
	//JIT_Execute_Init();
	MetaData_Init();
	//Heap_Init();
	//Finalizer_Init();
	//Socket_Init();
}

__global__
void Initialize_BCL2()
{
	//JIT_Execute_Init();
	//MetaData_Init();
	Type_Init();
	Heap_Init();
	Finalizer_Init();
	//Socket_Init();
}


function_space_specifier void* Bcl_Heap_Alloc(STRING assemblyName, STRING nameSpace, STRING name)
{
	tMD_TypeDef* type_def = MetaData_GetTypeDefFromFullName(assemblyName, nameSpace, name);
	void * result = Heap_AllocType(type_def);
	return result;
}