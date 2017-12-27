
#include <stdio.h>
#include "Type.h"
#include "Filesystem.h"

int main()
{
	printf("hi there\n");
	MetaData_Init();
	Type_Init();

	Gfs_add_file("a", "abc", 3);
	Gfs_add_file("b", "bbc", 3);
	int o1 = Gfs_open_file("a");
	size_t l1 = Gfs_length(o1);
	char * d1 = Gfs_read(o1);
	int o2 = Gfs_open_file("b");
	size_t l2 = Gfs_length(o2);
	char * d2 = Gfs_read(o2);
	return 0;
}
