
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
function_space_specifier void Initialize_BCL0(size_t size, size_t first_overhead, int count);

#ifdef __GNUC__
#include <execinfo.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

void handler(int sig) {

	Gprintf("Error: signal %d:\n", sig);
	//	Gprintf("Hi from handler!\n");
	// get void*'s for all entries on the stack
#ifdef __GNUC__
	void *array[100];
	int size = backtrace(array, 100);
	//	Gprintf("Hi from handler!\n");

	// print out all the frames to stderr
	backtrace_symbols_fd(array, size, STDERR_FILENO);
#endif
	Crash("segv");
	// Stop.
	exit(1);
}


function_space_specifier void CommonInitTheBcl(void * g, size_t size, size_t first_overhead, int count, struct _BCL_t ** pbcl)
{
//	Gprintf("Hi from setup!\n");
    signal(SIGSEGV, handler);   // install our handler

	// Erase the structure, then afterwards set everything up.
	struct _BCL_t * bcl = (struct _BCL_t*)g;
	*pbcl = bcl;
	_bcl_ = bcl;
	memset(bcl, 0, sizeof(struct _BCL_t));

	// basics/memory allocation.
	_bcl_->global_memory_heap = NULL;
	_bcl_->heap_list = NULL;
	_bcl_->kernel_base_index = 0;
	_bcl_->count = 0;

	// Init memory allocation.
	Initialize_BCL0(size, first_overhead, count);

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

function_space_specifier void InternalInitTheBcl(void * g, size_t size, size_t first_overhead, int count, void * s)
{
	CommonInitTheBcl(g, size, first_overhead, count, (struct _BCL_t**)s);
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
	U64 is_free;
	U64 size;
	struct header_t *next;
	struct header_t *prev;
	unsigned char data[1];
	// unsigned char pad_before[32];
	// real data
	// unsigned char pad_after[32];
};

function_space_specifier int roundUp(int numToRound, int multiple)
{
	return ((numToRound + multiple - 1) / multiple) * multiple;
}

function_space_specifier void Initialize_BCL0(size_t size, size_t first_overhead, int count)
{
	// Initialize heaps for BCL allocation.
	// Nothing can be done until this is done.
	// Layout, illustrated below, contains sections that are all fixed in size.
	//
	//  g points here:
	//  ======================================================================
	//           The struct _BCL_t
	//  ----------------------------------------------------------------------
	//  g+0x1000 Heap list-- count number of them.
	//           This is an array of the start of all heaps. Each heap is
	//           basically self contained, with allocated and free lists,
	//           which are per-thread oriented.
	//  ======================================================================

	// Each heap has the following structure:
	//  ----------------------------------------------------------------------
	//  Allocated list.
	//           This should be large enough as to handle all buffer pointers
	//           allocated by this package. The size is specified during set up.
	//           If zero or full, then debugging of heap pointers is limited.
	//           Allocation will still occur if full, but you can't track issues.
	//  -----------------------------------------------------------------------
	//  Free list.
	//           The list is a linked list of the free blocks, which vary in
	//           size. Free blocks are filled with 0xde bytes, and must be large
	//           enough. The algorithm is first fit.
	//           https://web.archive.org/web/20060409094110/http://www.osdcom.info/content/view/31/39/
	//  ======================================================================
	
	unsigned char * begin = (unsigned char *)_bcl_;
	unsigned char * end = begin + size;

//	Gprintf("beginning bcl 0x%08llx\n", begin);
//	Gprintf("out of bounds end should be 0x%08llx\n", end);

	_bcl_->count = count;
	_bcl_->padding = 256;
	_bcl_->pointer_count = 0x10;

	int size_for_bcl = roundUp(sizeof(struct _BCL_t), 0x1000);

	_bcl_->heap_list = (void**)(
			size_for_bcl
			+ (unsigned char*)_bcl_);
	
	unsigned long long int rest = size - size_for_bcl;

//	Gprintf("heap pointers array starts at 0x%08llx\n", _bcl_->heap_list);
//	Gprintf("rest 0x%08llx\n", rest);

	int size_for_heap_list = roundUp(sizeof(void*) * count, 0x1000);

	// Set up heap table with start of each heap.
	unsigned char * start = (unsigned char *)(
		size_for_bcl
		+ size_for_heap_list
		+ (unsigned char*)_bcl_);

	rest = rest - size_for_heap_list;
//	Gprintf("start at 0x%08llx\n", start);
//	Gprintf("rest 0x%08llx\n", rest);

	size_t second_and_on_heap_size = 0;
	if (count > 1)
		second_and_on_heap_size = (size - size_for_bcl - size_for_heap_list - first_overhead) / (count - 1);

//	Gprintf("The size of first heap is 0x%08llx\n", first_overhead);
//	Gprintf("The before size of remaining is 0x%08llx\n", second_and_on_heap_size);

	// make sure size is multiple of eight, so remove lower bits.
	second_and_on_heap_size = (second_and_on_heap_size >> 3) << 3;

//	Gprintf("The after size of remaining is 0x%08llx\n", second_and_on_heap_size);

	// check.
	if (size_for_bcl + size_for_heap_list + first_overhead + second_and_on_heap_size * (count - 1) > size)
		Crash("Miscomputed sizes.");

	// Set up start of individual heaps.
	for (int c = 0; c < count; ++c)
	{
//		Gprintf("start of heap %d at 0x%08llx\n", c, start);
		size_t heap_size = (c == 0) ? first_overhead : second_and_on_heap_size;
		if (start < begin)
			Crash("Ptr of heap %d starts before begin, 0x%08llx\n", c, start);
		if (start >= end)
			Crash("Ptr of heap %d ends after end, 0x%08llx\n", c, start);
		rest = rest - heap_size;
//		Gprintf("rest 0x%08llx\n", rest);
		_bcl_->heap_list[c] = start;
		start = start + heap_size;
	}

	// For each heap, set up allocated and free block list.
    for (int c = 0; c < count; ++c)
	{
		size_t heap_size = (c == 0) ? first_overhead : second_and_on_heap_size;
		unsigned char * s = (unsigned char *)_bcl_->heap_list[c];
		// allocate table for allocated.
		struct header_t ** f = (struct header_t **)s;
		struct header_t ** a = (struct header_t **)(s + _bcl_->pointer_count * sizeof(struct header_t*));
		struct header_t ** start_of_free_blocks = (struct header_t **)(s + _bcl_->pointer_count * sizeof(struct header_t*) * 2);
		for (int j = 0; j < _bcl_->pointer_count; ++j)
		{
			f[j] = NULL;
			a[j] = NULL;
		}

		// Set up pointer to first, and currently only, free block.
		struct header_t* fb = (struct header_t*)start_of_free_blocks;
		f[0] = fb;

		//Gprintf("ptr to begin of free block = 0x%08llx\n", (unsigned char *)fb);

		// Set up size of the free block in this heap.
		// header_overhead is # bytes of struct header_t up to "data".
		int header_overhead = (long long)(&fb->data[0]) - (long long)(&fb->is_free);
		_bcl_->head_size = header_overhead;

		// ptr_tables is the size of recorded free and used blocks.
		size_t ptr_tables = (size_t)(((long long)start_of_free_blocks) - (long long)f);

		// set up free block.
		fb->is_free = 1;
		fb->next = NULL;
		fb->prev = NULL;
		fb->size =
			heap_size			// all of the heap
			- ptr_tables	    // less the size for pointer tables free and used.
			- header_overhead	// less the overhead of header_t to data.
			- _bcl_->padding			// less padding after data for overrun checks.
			;

//		Gprintf("ptr to begin of free block = 0x%08llx size after header = 0x%08llx\n", ((unsigned char *)&fb->data[0]), fb->size);
//		Gprintf("padding %d\n", _bcl_->padding);
		
		// Set padding of free block.
		unsigned char * pad = ((unsigned char *)&fb->data[0]) + fb->size;

		if (pad < begin)
			Crash("Pad of heap %d starts before begin, 0x%08llx\n", c, pad);
		if (pad > end)
			Crash("Pad of heap %d ends after end, 0x%08llx\n", c, pad);
		if ((pad + _bcl_->padding) > end)
			Crash("Pad of heap end %d ends after end, 0x%08llx\n", c, pad + _bcl_->padding);

		memset(pad, 0xde, _bcl_->padding);
//		Gprintf("OK\n");
	}

	// check each free is not overlapping.
	// check end of fb is not over edge of whole heap.
	for (int c = 0; c < count; ++c)
	{
		size_t heap_size = (c == 0) ? first_overhead : second_and_on_heap_size;
		unsigned char * s = (unsigned char *)_bcl_->heap_list[c];
		unsigned char * end = s + heap_size;
		struct header_t ** f = (struct header_t **)s;
		struct header_t* fb = f[0];
		if (((unsigned char*)fb) + fb->size + _bcl_->padding - 1 >= end)
			Crash("Allocation of heaps off.");
	}
}

function_space_specifier void check_heap_structures()
{
	if (_bcl_ == NULL)
		Crash("BCL base pointer is null.");

	if (!(_bcl_->options & BCL_DEBUG_CHECK_HEAPS))
		return;

	for (int i = 0; i < _bcl_->count; ++i)
	{
		// Check free list for basic integrity.
		struct header_t ** f = (struct header_t **)_bcl_->heap_list[i];
		struct header_t ** a = f + _bcl_->pointer_count;
		struct header_t * curr = f[0];
		while (curr)
		{
			if (!(curr->is_free == 0 || curr->is_free == 1))
				Crash("BCL heap chain corrupt.");
			curr = curr->next;
		}
		// Verify every block here is on free list.
		for (int j = 0; j < _bcl_->pointer_count; ++j)
		{
			struct header_t * check = f[j];
			if (check == NULL) continue;
			//Gprintf("Check is 0x%08llx\n", check);
			curr = f[0];
			while (curr)
			{
				if (check == curr)
					break;
				curr = curr->next;
			}
			if (curr == NULL && check != NULL)
				Crash("Block in check list but not on free list.");
		}
		// Verify all free blocks have integrity padding unchanged.
		curr = f[0];
		while (curr)
		{
			unsigned char * pad = (unsigned char *)&curr->data + curr->size;
			memset(pad, 0xde, _bcl_->padding);
			for (int j = 0; j < _bcl_->padding; ++j)
			{
				if (pad[j] != 0xde)
					Crash("Padding overwrite.");
			}
			curr = curr->next;
		}
		// Verify all allocated blocks have integrity padding unchanged.
		curr = a[0];
		while (curr)
		{
			unsigned char * pad = (unsigned char *)&curr->data + curr->size;
			memset(pad, 0xde, _bcl_->padding);
			for (int j = 0; j < _bcl_->padding; ++j)
			{
				if (pad[j] != 0xde)
					Crash("Padding overwrite.");
			}
			curr = curr->next;
		}
	}
}

function_space_specifier void InternalCheckHeap()
{
	check_heap_structures();
}



function_space_specifier void * simple_malloc(size_t size)
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

	//Gprintf("Starting alloc, first a check...\n");
	check_heap_structures();

	size_t total_size;
	void *block;
	struct header_t *new_block;
	if (!size)
		return NULL;

	size_t before = size;
	size = roundUp(size, 8);
	//Gprintf("Block size was %llx, now %llx\n", before, size);
	struct header_t ** f = (struct header_t **)_bcl_->heap_list[threadId];
	// First fit algorithm of free list.
	//Gprintf("Looking for a free block\n");
	struct header_t * curr = f[0];
	while (curr)
	{
		if (curr->is_free && curr->size >= size)
		{
			//Gprintf("OK. Found free block 0x%08llx size %x\n", curr, curr->size);
			new_block = curr;
			break;
		}
		curr = curr->next;
	}

	if (new_block)
	{
		//Gprintf("got block, checking whether to split or use whole.\n");

		size_t old_size = new_block->size;
		size_t new_free_block_size =
			old_size
			- _bcl_->head_size;

		struct header_t * new_free_block = NULL;

		// split block if big enough.
		if (new_free_block_size > 8)
		{
			//Gprintf("split\n");

			// set up new free block.
			new_free_block =
				(struct header_t *)(&new_block->data[0]
					+ size
					+ _bcl_->padding);

			//Gprintf("New free block starts here 0x%08llx\n", new_free_block);
			//Gprintf("allocated block starts here 0x%08llx which will be reformatted.\n", new_block);

			// set up free block.
			new_free_block->is_free = 1;
			new_free_block->next = new_block->next;
			new_free_block->prev = new_block->prev;
			new_free_block->size = new_free_block_size;

			//Gprintf("Zapping the following addrs:\n");
			//Gprintf("0xde's, starting at 0x%08llx, and going for %d bytes\n",
			//	&new_block->data[0] + size, _bcl_->padding);
			//Gprintf("0's, starting at 0x%08llx, and going for %lld bytes\n",
			//		&new_block->data[0], size);
			// change allocated block.
			new_block->size = size;
			memset(&new_block->data[0] + size, 0xde, _bcl_->padding);
			memset(&new_block->data[0], 0x00, size);
		}
		else
		{
			//Gprintf("Using whole.\n");
		}

		// Set up free and allocated list.
		if (f[0] == new_block)
		{
			f[0] = new_free_block;
		}

		// Put allocated item on allocated list.
		struct header_t ** a = f + _bcl_->pointer_count;
		new_block->next = a[0];
		a[0] = new_block;

		//size_t free = 0;
		//struct header_t * curr2 = f[0];
		//while (curr2) {
		//	if (curr2->is_free) free += (curr2->size);
		//	curr2 = curr2->next;
		//}
		//Gprintf("Memory of heap %d left %lld\n", threadId, free);
		//Gprintf("Finishing up, first a check...\n");
		check_heap_structures();

		return (void*)&new_block->data[0];
	}
	Crash("No memory left.");
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
	if (_bcl_)
		return _bcl_->kernel_base_index;
	else
		return 0;
}

global_space_specifier void set_kernel_base_index(int i)
{
	if (_bcl_)
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


function_space_specifier void InternalSetOptions(U64 options)
{
	if (_bcl_)
		_bcl_->options = options;
}
