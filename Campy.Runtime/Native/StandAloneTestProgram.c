
#include <stdio.h>
#include <stdlib.h>
#include "Type.h"
#include "Filesystem.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "CLIFile.h"

int main()
{
	void * g = malloc(0x10000000);
	struct _BCL_t * pbcl;
	Initialize_BCL_Globals(g, 0x100000000, 1, &pbcl);
	
	{
		int result;
		FILE *file;
		file = fopen("C:\\Users\\kenne\\Documents\\Campy2\\Campy.Runtime\\Corlib\\bin\\Debug\\net20\\corlib.dll", "rb");
		if (!file)
		{
			printf("Unable to open file!");
			return 1;
		}
		fseek(file, 0, SEEK_END);
		int fileLen = ftell(file);
		fseek(file, 0, SEEK_SET);
		char * buffer = (char *)malloc(fileLen + 1);
		if (!buffer)
		{
			fprintf(stderr, "Memory error!");
			fclose(file);
			return;
		}
		fread(buffer, fileLen, 1, file);
		fclose(file);

		Gfs_add_file("corlib.dll", buffer, fileLen, &result);
	}
	{
		int result;
		FILE *file;
		file = fopen("C:\\Users\\kenne\\Documents\\Campy2\\ConsoleApp4\\bin\\Debug\\ConsoleApp4.exe", "rb");
		if (!file)
		{
			printf("Unable to open file!");
			return 1;
		}
		fseek(file, 0, SEEK_END);
		int fileLen = ftell(file);
		fseek(file, 0, SEEK_SET);
		char * buffer = (char *)malloc(fileLen + 1);
		if (!buffer)
		{
			fprintf(stderr, "Memory error!");
			fclose(file);
			return;
		}
		fread(buffer, fileLen, 1, file);
		fclose(file);

		Gfs_add_file("ConsoleApp1.exe", buffer, fileLen, &result);
	}

	MetaData_Init();
	Type_Init();

	tMetaData *pTypeMetaData;
	pTypeMetaData = CLIFile_GetMetaDataForAssembly("ConsoleApp1.exe");

	return 0;
}
