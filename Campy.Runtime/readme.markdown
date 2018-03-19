Campy Base Class Layer for NVIDIA GPU
=====================================
This code is based on Dot Net Anywhere, a NET CIL runtime. That code base is now inactive, the Campy BCL for GPU
is very much active. The runtime is written in C, designed to be as small as possible. This runtime is designed to
be substituted for calls from all other NET runtimes, and JIT'ed into GPU machine code.

How To Build
------------
This code is built for Windows using Visual Studio 2017 15.4.x using NVIDIA GPU Toolkit 9.1.85. Do not use
other compilers and toolkits as they have not been tested, and/or are incompatible. The solution file for builds
is located in the directory above this: ../Campy.sln. This code can be compiled
and run for the x64 Windows CPU using VS 2017, using the solution file ../Project1.sln.

-------------------------------

**Supported Features**

The corlib currently implements the following .NET CLR and language features:

* Generics
* Garbage collection and finalization
* Weak references
* Full exception handling - try/catch/finally
* PInvoke; although it's not the most pleasant or fully-featured implementation possible, but it will work cross-platform without libffi
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

[1]: https://github.com/chrisdunelm/DotNetAnywhere
[2]: http://www.mono-project.com
[3]: http://freetype.org
[4]: http://en.wikipedia.org/wiki/Threaded_code
