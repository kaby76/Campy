using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Campy;

namespace ConsoleApp4
{
    public class IntListSet
    {
        public static void IntListSetT()
        {
            // List of ints.
            List<int> x = new List<int>();
            int n = 4;
            for (int i = 0; i < n; ++i) x.Add(0);
            Campy.Parallel.For(n,
                i =>
                {
                    x[i] = i;
                });
            for (int i = 0; i < n; ++i) if (x[i] != i)
                    throw new Exception();
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
            StartDebugging();
            IntListSet.IntListSetT();
        }
    }
}
