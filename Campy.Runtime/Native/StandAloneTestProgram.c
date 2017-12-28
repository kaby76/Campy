
#include <stdio.h>
#include "Type.h"
#include "Filesystem.h"
extern unsigned char Gdata[];

int main()
{
	int result;
	Gfs_add_file("corlib.dll", Gdata, 267250 + 14, &result);

	MetaData_Init();
	Type_Init();

	return 0;
}
