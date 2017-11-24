using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Campy;
using System.Numerics;
using Campy.Compiler;
using Campy.Graphs;

namespace ConsoleApp4
{

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
        }

        static void Main(string[] args)
        {
            StartDebugging();

            //int n = 4;

            //string[] data = new string[] { "hihihihi", "there" };
            //int n2 = data.Length;
            //Campy.Parallel.For(n2, i =>
            //{
            //    data[i] = data[i].Substring(0, 3);
            //});



            //int[] t1 = new int[n];
            //Campy.Parallel.For(n, i => t1[i] = i);
            //for (int i = 0; i < n; ++i) if (t1[i] != i) throw new Exception();

            double[] buffer = new double[] { 0.1 };
            Campy.Parallel.For(buffer.Length, k =>
            {
                buffer[k] = Math.Sin(buffer[k]);
            });


        }
    }
}
