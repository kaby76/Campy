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

__global__ void Gfs_init();
__global__ int Gfs_add_file(char * name, char * file, size_t length);
__global__ int Gfs_remove_file(char * name);
__global__ int Gfs_open_file(char * name);
__global__ int Gfs_close_file(int file);
__global__ char * Gfs_read(int file);
__global__ size_t Gfs_length(int file);
