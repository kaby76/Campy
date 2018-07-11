using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using Campy;
using System.Linq;

namespace ConsoleApp4
{

    class Program
    {
        public static void TestMethod1()
        {
            int v1 = 0;
            int v2 = 0;
            int v3 = 0;
            int n = 4;
            var t1 = new int[n];
            var t2 = new int[n];
            Campy.Parallel.For(n, i =>
            {
                if (t1 == null)
                {
                    v1 = 1;
                }
                if (t1 != null)
                {
                    v2 = 1;
                }
                if (t2 != null)
                {
                    v3 = 1;
                }
                if (i % 2 == 0)
                    t1[i] = i * 20;
                else
                    t1[i] = i * 30;
            });
            for (int i = 0; i < n; ++i)
                if (i % 2 == 0)
                {
                    if (t1[i] != i * 20) throw new Exception();
                }
                else
                {
                    if (t1[i] != i * 30) throw new Exception();
                }
        }

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
            Campy.Parallel.For(3, i =>
            {
                //System.Console.WriteLine(i); // no explicit conversion.
                System.Console.WriteLine(i.ToString()); // value converted explicitly in code.
            });
        }
    }
}
