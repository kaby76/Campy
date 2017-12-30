
#include <stdio.h>
#include <stdlib.h>
#include "Type.h"
#include "Filesystem.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "CLIFile.h"

extern unsigned char Gdata[];

int main()
{
	{
		int result;
		Gfs_add_file("corlib.dll", Gdata,
			/* see corlib.c file, number of lines with 50 bytes per line plus last partial line */
			5345 * 50 + 14, &result);
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
