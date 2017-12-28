#pragma once
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

__device__ void Gfs_init();
__device__ void Gfs_add_file(char * name, char * file, size_t length, int * result);
__device__ void Gfs_remove_file(char * name, int * result);
__device__ void Gfs_open_file(char * name, int * result);
__device__ void Gfs_close_file(int file, int * result);
__device__ void Gfs_read(int file, char ** result);
__device__ void Gfs_length(int file, size_t * result);
