using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Campy;
using System.Numerics;
using Campy.Compiler;

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
            //StartDebugging();
            var stopwatc = new Stopwatch();
            {
                stopwatc.Reset();
                stopwatc.Start();
                Extent ex = new Extent(3, 5); // three rows, five columns.
                int[,] b = new int[ex[0], ex[1]];
                for (int i = 0; i < ex[0]; ++i)
                for (int j = 0; j < ex[1]; ++j)
                    b[i, j] = j + i * ex[1];
                stopwatc.Stop();
            }
            {
                stopwatc.Reset();
                stopwatc.Start();
                Extent ex = new Extent(3, 5); // three rows, five columns.
                int[,] b = new int[ex[0], ex[1]];
                for (int d = 0; d < ex[0] * ex[1]; ++d)
                {
                    int i = d / ex[1];
                    int j = d % ex[1];
                    b[i, j] = d;
                }
                stopwatc.Stop();
            }
            {
                stopwatc.Reset();
                stopwatc.Start();
                Extent ex = new Extent(3, 5, 2);
                int[,,] b = new int[ex[0], ex[1], ex[2]];
                System.Console.WriteLine("ex0 " + ex[0]);
                System.Console.WriteLine("ex1 " + ex[1]);
                System.Console.WriteLine("ex2 " + ex[2]);
                for (int d = 0; d < ex[0] * ex[1] * ex[2]; ++d)
                {
                    // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
                    int i = d / (ex[1] * ex[2]);
                    int r = d % (ex[1] * ex[2]);
                    int j = r / ex[2];
                    int k = r % ex[2];
                    System.Console.WriteLine("d " + d + " i " + i + " j " + j + " k " + k);
                    b[i, j, k] = d;
                }
                stopwatc.Stop();
            }
            {
                Extent ex = new Extent(3, 5); // three rows, five columns.
                int[,] b = new int[ex[0], ex[1]];
                Campy.Parallel.For(15, d =>
                {
                    int i = d / ex[1];
                    int j = d % ex[1];
                    b[i, j] = d;
                });
            }
            {
                Extent ex = new Extent(3, 5, 2); // three rows, five columns.
                int[,,] b = new int[ex[0], ex[1], ex[2]];
                Campy.Parallel.For(ex[0] * ex[1] * ex[2], d =>
                {
                    int i = d / (ex[1] * ex[2]);
                    int r = d % (ex[1] * ex[2]);
                    int j = r / ex[2];
                    int k = r % ex[2];
                    b[i, j, k] = d;
                });
            }
            System.Console.WriteLine("cpu = " + stopwatc.Elapsed);
            
        }
    }
}
