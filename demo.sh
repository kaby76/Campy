#!/bin/bash
mkdir test
cd test
dotnet new console
cat - << HERE > Program.cs
namespace test
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = 4;
            int[] x = new int[n];
            Campy.Parallel.For(n, i => x[i] = i);
            for (int i = 0; i < n; ++i)
                System.Console.WriteLine(x[i]);
        }
    }
}
HERE
dotnet add package Campy
dotnet build
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)
	dotnet publish -r ubuntu.16.04-x64
	cd bin/Debug/netcoreapp2.0/ubuntu.16.04-x64/publish/
	;;
    Darwin*)
	echo Cannot target Mac yet.
	exit 1
	;;
    CYGWIN*)
	dotnet publish -r win10-x64
	cd bin/Debug/netcoreapp2.0/win10-x64/publish/
	;;
    MINGW*)
	dotnet publish -r win10-x64
	cd bin/Debug/netcoreapp2.0/win10-x64/publish/
	;;
    *)
	echo Unknown machine.
	exit 1
	;;
esac
./test.exe
echo Output should be four lines of integers, 0 to 3.
