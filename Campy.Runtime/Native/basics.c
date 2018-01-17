#include "Compat.h"
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
#include <cuda.h>


__device__  void gpuexit(int _Code) {}

__device__ size_t Gstrlen(
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


__device__ void * global_memory_heap;

struct header_t {
	struct header_t *next;
	struct header_t *prev;
	size_t size;
	unsigned is_free;
};

__device__ struct header_t* head;

__global__
void Initialize_BCL0(void * g, size_t size, int count)
{
	head = (struct header_t*)g;
	int size_for_headers = sizeof(struct header_t) * count;
	unsigned char * ptr = ((unsigned char *)head) + size_for_headers;
	long long s = (long long) ptr;
	long long e = s + size;
	int overhead = 16777216;
	int remainder = size - size_for_headers - overhead;
	int per_thread_remainder = size / (count - 1);
	per_thread_remainder = per_thread_remainder >> 3;
	per_thread_remainder = per_thread_remainder << 3;
	for (int c = 0; c < count; ++c)
	{
		struct header_t * start = &head[c];
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
			siz = overhead;
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
	printf("done\n");
}

__device__ struct header_t *get_free_block(size_t size)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;

	struct header_t *curr = &head[threadId];
	while (curr) {
		if (curr->is_free && curr->size >= size)
			return curr;
		curr = curr->next;
	}
	return NULL;
}

__device__ int roundUp(int numToRound, int multiple)
{
	return ((numToRound + multiple - 1) / multiple) * multiple;
}

__device__ void * simple_malloc(size_t size)
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


__device__ void* __cdecl Grealloc(
	void*  _Block,
	size_t _Size
)
{
	void * result = simple_malloc(_Size);
	memcpy(result, _Block, _Size);
	Gfree(_Block);
    return result;
}


__device__ void* Gmalloc(
	size_t _Size
)
{
	void * result = simple_malloc(_Size);
	return result;
}

__device__ void Gfree(
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


__device__
void* Bcl_Heap_Alloc(STRING assemblyName, STRING nameSpace, STRING name)
{
	tMD_TypeDef* type_def = MetaData_GetTypeDefFromFullName(assemblyName, nameSpace, name);
	void * result = Heap_AllocType(type_def);
	return result;
}