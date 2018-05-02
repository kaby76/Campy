
#include "_BCL_.h"
#include "Sys.h"
#include "MetaData.h"
#include "JIT.h"
#include "Type.h"
#include "Finalizer.h"
#include "Heap.h"
#include "System.Array.h"
#include "System.Net.Sockets.Socket.h"
#include "Gprintf.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

gpu_space_specifier struct _BCL_t * _bcl_;
function_space_specifier void Initialize_BCL0(void * g, size_t size, int count);


function_space_specifier void CommonInitTheBcl(void * g, size_t size, int count, struct _BCL_t ** pbcl)
{
	// Erase the structure, then afterwards set everything up.
	struct _BCL_t * bcl = (struct _BCL_t*)g;
	*pbcl = bcl;
	_bcl_ = bcl;
	memset(bcl, 0, sizeof(struct _BCL_t));

	// basics/memory allocation.
	_bcl_->global_memory_heap = NULL;
	_bcl_->head = NULL;
	_bcl_->kernel_base_index = 0;

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
	bcl->tableRowSize = (unsigned int *)malloc(MAX_TABLES * sizeof(unsigned int));

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

function_space_specifier void InternalInitTheBcl(void * g, size_t size, int count, void * s)
{
	CommonInitTheBcl(g, size, count, (struct _BCL_t**)s);
}

global_space_specifier void Initialize_BCL_Globals(void * g, size_t size, int count, struct _BCL_t ** pbcl)
{
	CommonInitTheBcl(g, size, count, pbcl);
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
}

function_space_specifier struct header_t *get_free_block(size_t size)
{
#ifdef  __CUDA_ARCH__
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

//	printf("simple_malloc %d\n", size);
//	printf("dump of entire blocks\n");
//	struct header_t *curr = &_bcl_->head[0];
//	while (curr) {
//		printf("curr %llx\n", curr);
//		printf("curr next %llx\n", curr->next);
//		printf("curr prev %llx\n", curr->prev);
//		printf("curr size %d\n", curr->size);
//		printf("curr free %d\n", curr->is_free);
//		curr = curr->next;
//	}
//	printf("------------------\n");

	if (header)
	{
//		printf("simple_malloc allocating\n");
		// split block if big enough.
		if (header->size > (size + sizeof(struct header_t)))
		{
//			printf("header big enough\n");
//			printf("header %llx\n", header);
//			printf("header next %llx\n", header->next);
//			printf("header prev %llx\n", header->prev);
//			printf("header size %d\n", header->size);
//			printf("header free %d\n", header->is_free);
			int original_size = header->size;
			int skip = size + sizeof(struct header_t);
//			printf("skip %d\n", skip);
			unsigned char * ptr = ((unsigned char *)header) + skip;
			struct header_t * new_free = (struct header_t *)ptr;
			new_free->is_free = 1;
			new_free->size = original_size - skip;
			new_free->prev = header->prev;
			new_free->next = header->next;
//			printf("header %llx\n", header);
//			printf("header next %llx\n", header->next);
//			printf("header prev %llx\n", header->prev);
//			printf("header size %d\n", header->size);
//			printf("header free %d\n", header->is_free);
//			printf("new_free %llx\n", new_free);
//			printf("new_free next %llx\n", new_free->next);
//			printf("new_free prev %llx\n", new_free->prev);
//			printf("new_free size %d\n", new_free->size);
//			printf("new_free free %d\n", new_free->is_free);
//			printf("----\n");
			if (new_free->prev != NULL)
			{
				new_free->prev->next = new_free;
//				printf("updated\n");
			}
			header->size = size;
//			printf("header next %llx\n", header->next);
//			printf("header prev %llx\n", header->prev);
//			printf("header size %d\n", header->size);
//			printf("header free %d\n", header->is_free);
//			printf("new_free %llx\n", new_free);
//			printf("new_free next %llx\n", new_free->next);
//			printf("new_free prev %llx\n", new_free->prev);
//			printf("new_free size %d\n", new_free->size);
//			printf("new_free free %d\n", new_free->is_free);
//			printf("++++\n");
		}
		header->is_free = 0;
//		printf("dump of entire blocks after\n");
//		struct header_t *curr2 = &_bcl_->head[0];
//		while (curr2) {
//			printf("curr %llx\n", curr2);
//			printf("curr next %llx\n", curr2->next);
//			printf("curr prev %llx\n", curr2->prev);
//			printf("curr size %d\n", curr2->size);
//			printf("curr free %d\n", curr2->is_free);
//			curr2 = curr2->next;
//		}
//		printf("------------------\n");
		return (void*)(header + 1);
	}
	return NULL;
}

function_space_specifier void* Grealloc(void*  _Block, size_t _Size)
{
	void * result = simple_malloc(_Size);
	memcpy(result, _Block, _Size);
	Gfree(_Block);
    return result;
}

function_space_specifier void* Gmalloc(size_t _Size)
{
	void * result = simple_malloc(_Size);
	return result;
}

function_space_specifier void Gfree(void*  _Block)
{
}

function_space_specifier void InternalInitializeBCL1()
{
	MetaData_Init();
}

global_space_specifier void Initialize_BCL1()
{
	MetaData_Init();
}

function_space_specifier void InternalInitializeBCL2()
{
	Type_Init();
	Heap_Init();
	Finalizer_Init();
}

global_space_specifier void Initialize_BCL2()
{
	Type_Init();
	Heap_Init();
	Finalizer_Init();
}


function_space_specifier void* Bcl_Array_Alloc(tMD_TypeDef* element_type_def, int rank, unsigned int* lengths)
{
	tMD_TypeDef* array_type_def = Type_GetArrayTypeDef(element_type_def, NULL, NULL);
	return (void*)SystemArray_NewVector(array_type_def, rank, lengths);
}

function_space_specifier int get_kernel_base_index()
{
	return _bcl_->kernel_base_index;
}

global_space_specifier void set_kernel_base_index(int i)
{
	_bcl_->kernel_base_index = i;
}


//function_space_specifier void store_static_field(char * type, char * field)
//{
//	tMD_FieldDef *pFieldDef;
//	tMD_TypeDef *pParentType;
//
//	pFieldDef = (tMD_FieldDef*)GET_OP();
//	pParentType = pFieldDef->pParentType;
//	// Check that any type (static) constructor has been called
//	if (pParentType->isTypeInitialised == 0) {
//		// Set the state to initialised
//		pParentType->isTypeInitialised = 1;
//		// Initialise the type (if there is a static constructor)
//		if (pParentType->pStaticConstructor != NULL) {
//			tMethodState *pCallMethodState;
//
//			// Call static constructor
//			// Need to re-run this instruction when we return from static constructor call
//			//pCurrentMethodState->ipOffset -= 2;
//			pCurOp -= 2;
//			pCallMethodState = MethodState_Direct(pThread, pParentType->pStaticConstructor, pCurrentMethodState, 0);
//			// There can be no parameters, so don't need to set them up
//			CHANGE_METHOD_STATE(pCallMethodState);
//			GO_NEXT_CHECK();
//		}
//	}
//	if (op == JIT_LOADSTATICFIELD_CHECKTYPEINIT_F64) {
//		U64 value;
//		value = *(U64*)(pFieldDef->pMemory);
//		PUSH_U64(value);
//	}
//	else if (op == JIT_LOADSTATICFIELD_CHECKTYPEINIT_VALUETYPE) {
//		PUSH_VALUETYPE(pFieldDef->pMemory, pFieldDef->memSize, pFieldDef->memSize);
//	}
//	else {
//		U32 value;
//		if (op == JIT_LOADSTATICFIELDADDRESS_CHECKTYPEINIT) {
//			value = (U32)(pFieldDef->pMemory);
//		}
//		else {
//			value = *(U32*)pFieldDef->pMemory;
//		}
//		PUSH_U32(value);
//	}
//}