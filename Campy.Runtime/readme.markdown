Campy Base Class Layer for NVIDIA GPU
=====================================
This code is the Campy Base Class Layer (BCL) for the NVIDIA GPU. It is
based on Dot Net Anywhere, a thin NET 2.0 CIL runtime, which is now inactive.
The runtime is written in C# and CUDA C. While most of the runtime can be implemented in C#,
some parts must be implemented in CUDA C in order to access the NVIDIA CUDA runtime library.
The Campy BCL is designed to be as small as possible, supporting
and supporting computing on the GPU. Calls to the NET runtime
is substituted for calls from all other NET runtimes, and JIT'ed into GPU machine code.

How to build the runtime
-------------------------
This code is built for Windows using Visual Studio 2017 15.4.x using NVIDIA GPU Toolkit 9.1.85. Do not use
other compilers and toolkits as they have not been tested, and/or are incompatible. The solution file for builds
is located in the directory above this: ../Campy.sln. This code can be compiled
and run for the x64 Windows CPU using VS 2017, using the solution file ../Project1.sln.

-------------------------------

**Supported Features**

The core library implements the following .NET CLR and language features:

* Generics
* Garbage collection and finalization
* Weak references
* Interfaces
* Delegates
* Events
* Nullable types
* Single-dimensional arrays
* Multi-threading; not using native threads, only actually runs in one native thread
* Very limited read-only reflection; typeof(), .GetType(), Type.Name, Type.Namespace, Type.IsEnum(), \<object\>.ToString() only

**Unsupported features**

* Attributes
* Most reflection
* Multi-dimensional arrays
* Unsafe code
* PInvoke; although it's not the most pleasant or fully-featured implementation possible, but it will work cross-platform without libffi
* Full exception handling - try/catch/finally

[1]: https://github.com/chrisdunelm/DotNetAnywhere
[2]: http://www.mono-project.com
[3]: http://freetype.org
[4]: http://en.wikipedia.org/wiki/Threaded_code
