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

function_space_specifier static char** names;
function_space_specifier static char** files;
function_space_specifier static size_t* lengths;
function_space_specifier static boolean init;
function_space_specifier static int initial_size;

__global__ void Bcl_Gfs_init()
{
	Gfs_init();
}

function_space_specifier void Gfs_init()
{
	printf("Gfs_init in\n");
	initial_size = 10;
	names = (char**)Gmalloc(initial_size * sizeof(char*));
	memset(names, 0, initial_size * sizeof(char*));
	files = (char**)Gmalloc(initial_size * sizeof(char*));
	memset(files, 0, initial_size * sizeof(char*));
	lengths = (size_t*)Gmalloc(initial_size * sizeof(size_t));
	memset(lengths, 0, initial_size * sizeof(size_t));
	init = 1;
	printf("Gfs_init out\n");
}

__global__ void Bcl_Gfs_add_file(char * name, char * file, size_t length, int * result)
{
	Gfs_add_file(name, file, length, result);
}

function_space_specifier void Gfs_add_file(char * name, char * file, size_t length, int * result)
{
	printf("name %s\n", name);
	if (init == 0) Gfs_init();
	char ** ptr_name = names;
	char ** ptr_file = files;
	size_t * ptr_length = lengths;
	for (int i = 0; i < initial_size; ++i)
	{
		if (*ptr_name == NULL)
		{
			printf("name slot null, adding\n");
			*ptr_name = Gstrdup(name);
			*ptr_file = (char *)Gmalloc(length);
			memcpy(*ptr_file, file, length);
			printf("copy\n");
			*ptr_length = length;
			printf("len %d\n", length);
			printf("returning %d\n", i);
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

__global__ void Bcl_Gfs_remove_file(char * name, int * result)
{
	Gfs_remove_file(name, result);
}

function_space_specifier void Gfs_remove_file(char * name, int * result)
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
			Gfree(*ptr_name);
			*ptr_name = NULL;
			Gfree(*ptr_file);
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

__global__ void Bcl_Gfs_open_file(char * name, int * result)
{
	Gfs_open_file(name, result);
}

function_space_specifier void Gfs_open_file(char * name, int * result)
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

__global__ void Bcl_Gfs_close_file(int file, int * result)
{
	Gfs_close_file(file, result);
}

function_space_specifier void Gfs_close_file(int file, int * result)
{
	if (init == 0) Gfs_init();
	*result = 0;
}

__global__ void Bcl_Gfs_read(int file, char ** result)
{
	Gfs_read(file, result);
}

function_space_specifier void Gfs_read(int file, char ** result)
{
	if (init == 0) Gfs_init();
	*result = files[file];
}

__global__ void Bcl_Gfs_length(int file, size_t * result)
{
	Gfs_length(file, result);
}

function_space_specifier void Gfs_length(int file, size_t * result)
{
	if (init == 0) Gfs_init();
	*result = lengths[file];
}

