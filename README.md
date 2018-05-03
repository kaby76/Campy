# Campy

This project is an API for GP-GPU computing in .NET languages. With Campy, one writes GP-GPU code
without the usual boilerplate code you see in CUDA/OpenCL, but instead use a simplified model of a multiprocessor GPU/CPU computer,
freeing the developer to focus exclusively on the algorithm. 
The API compiles and runs CIL code into native GPU code using LLVM. Supported are value types,
reference types, methods, generics, lambdas, delegates, and closures. Other C#/GPU projects exist,
but do not offer a clean, boilerplate-free interface, nor support C# beyond value types. Campy provides
a base class library (BCL) for the GPU, implementing strings, arrays, generics, exceptions, delegates,
dynamically memory allocation, reflection, and native calls.

This project is in the early stage of development. Releases are currently for demonstration purposes.
The only available method is essentially Campy.Parallel.For(), and it will undergo wild changes in signature
as I refine the programming model. There are also several debugging output
switches (see example below).

~~~~
public static void Campy.Parallel.For(int number_of_threads, _Kernel_type kernel)
public delegate void Campy.Types._Kernel_type(Index idx);
~~~~

# Targets

* Windows 10 (x64), Ubuntu 16.04 (x64), Net Framework >= 4.6.1, Net Core 2.0; CUDA GPU Toolkit 9.1.85; Maxwell or better GPU.

# Campy under a minute #
(Make sure to install Net Core 2.0, https://www.microsoft.com/net/learn/get-started/windows.)
~~~~
mkdir test
cd test
dotnet new console
dotnet add package Campy
# copy the example code below into test.cs.
dotnet build
dotnet publish -r win10-x64
# for Ubuntu, dotnet publish -r ubuntu.16.04-x64
cd bin/Debug/netcoreapp2.0/win10-x64/publish/
./test.exe
# output should be four lines of integers, 0 to 3.
~~~~

# Example #

~~~~
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
~~~~

Additional examples in Campy test area (https://github.com/kaby76/Campy/tree/master/Tests), including Reduction, various sorting algorithms, FFT, etc.
 
