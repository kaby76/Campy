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

        bool ApproxEqual(double a, double b)
        {
            if (b > a)
                return (b - a) < 0.01;
            else
                return (a - b) < 0.01;
        }


        static void Main(string[] args)
        {
            StartDebugging();
            double[] buffer = new double[] { 0.1, 0.2, 0.3 };
            Campy.Parallel.For(buffer.Length, k =>
            {
                buffer[k] = Math.Sin(buffer[k]);
            });
        }
    }
}
