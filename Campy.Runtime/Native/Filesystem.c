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
#include "Filesystem.h"

static char** names;
static char** files;
static size_t* lengths;
static boolean init;
static int initial_size;

__global__ void Gfs_init()
{
	initial_size = 10;
	names = malloc(initial_size * sizeof(char*));
	memset(names, 0, initial_size * sizeof(char*));
	files = malloc(initial_size * sizeof(char*));
	memset(files, 0, initial_size * sizeof(char*));
	lengths = malloc(initial_size * sizeof(size_t));
	memset(lengths, 0, initial_size * sizeof(size_t));
	init = 1;
}

__global__ int Gfs_add_file(char * name, char * file, size_t length)
{
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name == NULL)
		{
			*ptr_name = strdup(name);
			*ptr_file = malloc(length);
			memcpy(*ptr_file, file, length);
			*ptr_length = length;
			return 1;
		}
		else
		{
			ptr_name++;
			ptr_file++;
			ptr_length++;
		}
	}
	return 0;
}

__global__ int Gfs_remove_file(char * name)
{
	// Delete a pseudo file system for the meta system to work,
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name != NULL && strcmp(*ptr_name, name) == 0)
		{
			free(*ptr_name);
			*ptr_name = NULL;
			free(*ptr_file);
			*ptr_file = NULL;
			*ptr_length = 0;
			return 1;
		}
		else
		{
			ptr_name++;
			ptr_file++;
			ptr_length++;
		}
	}
	return 0;
}

__global__ int Gfs_open_file(char * name)
{
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name != NULL && strcmp(*ptr_name, name) == 0)
		{
			return i;
		}
		else
		{
			ptr_name++;
			ptr_file++;
			ptr_length++;
		}
	}
	return -1;
}

__global__ int Gfs_close_file(int file)
{
	if (init == 0) Gfs_init();
	return 0;
}

__global__ char * Gfs_read(int file)
{
	if (init == 0) Gfs_init();
	return files[file];
}

__global__ size_t Gfs_length(int file)
{
	if (init == 0) Gfs_init();
	return lengths[file];
}

