
#include <stdio.h>
#include <stdlib.h>
#include "Type.h"
#include "Filesystem.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "CLIFile.h"

int main(int argc, char ** argv)
{
	void * g = malloc(0x10000000);
	struct _BCL_t * pbcl;
	Initialize_BCL_Globals(g, 0x100000000, 1, &pbcl);

	for (int i = 0; i < argc - 1; ++i)
	{
		int result;
		FILE *file;
		file = fopen(argv[i+1], "rb");
		if (!file)
		{
			printf("Unable to open file!");
			continue;
		}
		fseek(file, 0, SEEK_END);
		int fileLen = ftell(file);
		fseek(file, 0, SEEK_SET);
		char * buffer = (char *)malloc(fileLen + 1);
		if (!buffer)
		{
			fprintf(stderr, "Memory error!");
			fclose(file);
			continue;
		}
		fread(buffer, fileLen, 1, file);
		fclose(file);
		char * fn = strdup(argv[i + 1]);
		int w = -1;
		for (int j = 0; fn[j]; ++j)
			if (fn[j] == '/' || fn[j] == '\\') w = j;
		if (w >= 0)
		{
			int j;
			for (j = w + 1; fn[j]; ++j) fn[j - w - 1] = fn[j];
			fn[j - w - 1] = 0;
		}
		Gfs_add_file(fn, buffer, fileLen, &result);
	}

	if (argc - 1 > 0)
	{
		MetaData_Init();
		Type_Init();
	}

	for (int i = 0; i < argc - 1; ++i)
	{
		tMetaData *pTypeMetaData;
		char * fn = strdup(argv[i + 1]);
		int w = -1;
		for (int j = 0; fn[j]; ++j)
			if (fn[j] == '/' || fn[j] == '\\') w = j;
		if (w >= 0)
		{
			int j;
			for (j = w + 1; fn[j]; ++j) fn[j - w - 1] = fn[j];
			fn[j - w - 1] = 0;
		}
		pTypeMetaData = CLIFile_GetMetaDataForAssembly(fn);
	}

	return 0;
}
