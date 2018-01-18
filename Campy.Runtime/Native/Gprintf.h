#pragma once

#if defined(CUDA)
#include <cuda.h>
#endif

#include <stdarg.h>
#define LDOUBLE double
#define LLONG long
#define sizeret_t int
#define P_CONST const
#define my_modf modf
#define NULLP (0)

function_space_specifier sizeret_t Gvsnprintf(char *str, size_t count, P_CONST char *fmt, va_list args);
function_space_specifier sizeret_t Gvasprintf(char **ptr, P_CONST char *format, va_list ap);
function_space_specifier sizeret_t Gasprintf(char **ptr, P_CONST char *format, ...);
function_space_specifier int Gvsprintf(char *buf, const char *format, va_list ap);
function_space_specifier int Gsprintf(char* const _Buffer, char const* const _Format, ...);
function_space_specifier int Gprintf(const char *format, ...);
function_space_specifier int Gvprintf(const char * format, va_list arg);
