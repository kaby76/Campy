#pragma once

#include <cuda.h>
#include <stdarg.h>
#define LDOUBLE double
#define LLONG long
#define sizeret_t int
#define P_CONST const
#define my_modf modf
#define NULLP (0)

__device__ sizeret_t Gvsnprintf(char *str, size_t count, P_CONST char *fmt, va_list args);
__device__ sizeret_t Gvasprintf(char **ptr, P_CONST char *format, va_list ap);
__device__ sizeret_t Gasprintf(char **ptr, P_CONST char *format, ...);
__device__ int Gvsprintf(char *buf, const char *format, va_list ap);
__device__ int Gsprintf(char* const _Buffer, char const* const _Format, ...);
