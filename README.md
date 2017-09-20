# Campy

This project is an API for GP-GPU computing in .NET languages. With Campy, one writes GP-GPU code
free of the usual boilerplate code, freeing the developer to focus exclusively on the algorithm.
The API compiles and runs CIL code into native GPU code using LLVM. Supported are value types,
reference types, methods, generics, lambdas, delegates, and closures. Other C#/GPU projects exist,
but do not offer a clean, boilerplate-free interface, supporting C# beyond value types.

This project is in the early stage of development. Releases are currently for demonstration purposes.
The only available method is essentially Campy.Parallel.For(), and it will undergo wild changes in signature
as I refine the programming model. There are also several debugging output
switches (see example below).

~~~~
public static void Campy.Parallel.For(int number_of_threads, _Kernel_type kernel)
public delegate void Campy.Types._Kernel_type(Index idx);
~~~~

# Targets

* Windows 10 (x64), Net Framework.

#### Net Framework App on Windows

Use the Package Manager GUI in VS 2017 to add in the package "Campy". Or,
download the package from NuGet (https://www.nuget.org/packages/Campy) and
add the package "Campy" from the nuget package manager console.

Set up the build of your C# application with Platform = "AnyCPU", Configuration = "Debug" or "Release". In the Properties for the
application, either un-check "Prefer 32-bit" if you want to run as 64-bit app, or checked if you want to run as a 32-bit app.

# Example #

~~~~
using System.Collections.Generic;
using System.Linq;

namespace ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //Campy.Utils.Options.Set("graph_trace", true);
            //Campy.Utils.Options.Set("module_trace", true);
            //Campy.Utils.Options.Set("name_trace", true);
            //Campy.Utils.Options.Set("cfg_construction_trace", true);
            //Campy.Utils.Options.Set("dot_graph", true);
            //Campy.Utils.Options.Set("jit_trace", true);
            //Campy.Utils.Options.Set("memory_trace", true);

            int max_level = 16;
            int n = Bithacks.Power2(max_level);
            List<int> data = Enumerable.Repeat(0, n).ToList();

            Campy.Parallel.For(n, idx => data[idx] = 1);
            for (int level = 1; level <= Bithacks.Log2(n); level++)
            {
                int step = Bithacks.Power2(level);
                Campy.Parallel.For(n / step, idx =>
                {
                    var i = step * idx;
                    data[i] = data[i] + data[i + step / 2];
                });
            }
            System.Console.WriteLine("sum = " + data[0]);
        }
    }
}
~~~~
