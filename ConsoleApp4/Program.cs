using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Campy;

namespace ConsoleApp4
{
    class B
    {
        public virtual int Yo()
        {
            return 1;
        }
    }

    class A : B
    {
        public override int Yo()
        {
            return 2;
        }
    }

    class Program
    {
        static void StartDebugging()
        {
            Campy.Utils.Options.Set("graph_trace");
            Campy.Utils.Options.Set("module_trace");
            Campy.Utils.Options.Set("name_trace");
            Campy.Utils.Options.Set("cfg_construction_trace");
            Campy.Utils.Options.Set("dot_graph");
            Campy.Utils.Options.Set("jit_trace");
            Campy.Utils.Options.Set("memory_trace");
            Campy.Utils.Options.Set("ptx_trace");
            Campy.Utils.Options.Set("state_computation_trace");
            Campy.Utils.Options.Set("continue_with_no_resolve");
            Campy.Utils.Options.Set("copy_trace");
            Campy.Utils.Options.Set("runtime_trace");
        }

        static void Main(string[] args)
        {
            var a = new A();
            var b = new B();
            Campy.Parallel.Compile(typeof(A));
            Campy.Parallel.Compile(typeof(B));
            B c = a;
            System.Console.WriteLine(c.Yo());
            StartDebugging();
            int[] xx = new int[4];
            Parallel.For(4, i =>
            {
                xx[i] = c.Yo();
            });
        }
    }
}
