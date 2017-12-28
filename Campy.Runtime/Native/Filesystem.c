// GFS -- a pseudo file system for the GPU.
// This file system is for the GPU BCL meta system. There are
// only a few functions available, on it's rather limited in scope.
//
// The file system is:
// * completely flat. There are no directories.
// * completely in memory.
// * an association of a name (null-terminated char string) with
//      a byte array of a given length.
//
// Operations:
// Gfs_init()
// Gfs_add_file()
// Gfs_remove_file()
// Gfs_open_file()
// Gfs_close_file()
// Gfs_read()
// Gfs_length()

#include "Compat.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "Gstring.h"
#include "Filesystem.h"

__device__ static char** names;
__device__ static char** files;
__device__ static size_t* lengths;
__device__ static boolean init;
__device__ static int initial_size;

__device__ void Gfs_init()
{
	initial_size = 10;
	names = (char**)malloc(initial_size * sizeof(char*));
	memset(names, 0, initial_size * sizeof(char*));
	files = (char**)malloc(initial_size * sizeof(char*));
	memset(files, 0, initial_size * sizeof(char*));
	lengths = (size_t*)malloc(initial_size * sizeof(size_t));
	memset(lengths, 0, initial_size * sizeof(size_t));
	init = 1;
}

__device__ void Gfs_add_file(char * name, char * file, size_t length, int * result)
{
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name == NULL)
		{
			*ptr_name = Gstrdup(name);
			*ptr_file = (char *)malloc(length);
			memcpy(*ptr_file, file, length);
			*ptr_length = length;
			*result = i;
			return;
		}
		else
		{
			ptr_name++;
			ptr_file++;
			ptr_length++;
		}
	}
	*result = -1;
}

__device__ void Gfs_remove_file(char * name, int * result)
{
	// Delete a pseudo file system for the meta system to work,
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name != NULL && Gstrcmp(*ptr_name, name) == 0)
		{
			free(*ptr_name);
			*ptr_name = NULL;
			free(*ptr_file);
			*ptr_file = NULL;
			*ptr_length = 0;
			*result = i;
			return;
		}
		else
		{
			ptr_name++;
			ptr_file++;
			ptr_length++;
		}
	}
	*result = -1;
}

__device__ void Gfs_open_file(char * name, int * result)
{
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name != NULL && Gstrcmp(*ptr_name, name) == 0)
		{
			*result = i;
			return;
		}
		else
		{
			ptr_name++;
			ptr_file++;
			ptr_length++;
		}
	}
	*result = -1;
}

__device__ void Gfs_close_file(int file, int * result)
{
	if (init == 0) Gfs_init();
	*result = 0;
}

__device__ void Gfs_read(int file, char ** result)
{
	if (init == 0) Gfs_init();
	*result = files[file];
}

__device__ void Gfs_length(int file, size_t * result)
{
	if (init == 0) Gfs_init();
	*result = lengths[file];
}

