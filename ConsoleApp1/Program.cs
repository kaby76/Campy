using System;
using System.Collections.Generic;
using Campy;

namespace ConsoleApp1
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
            Campy.Utils.Options.Set("copy_trace");
            Campy.Utils.Options.Set("runtime_trace");
        }

        static void Main(string[] args)
        {
            StartDebugging();
            //{
            //    int ex0 = 3;
            //    int ex1 = 5;
            //    // three rows, five columns.
            //    int[,] b = new int[ex0, ex1];
            //    for (int d = 0; d < ex0 * ex1; ++d)
            //    {
            //        int i = d / ex1;
            //        int j = d % ex1;
            //        b[i, j] = d;
            //    }

            //    int[,] c = new int[ex0, ex1];
            //    Campy.Parallel.For(15, d =>
            //    {
            //        int i = d / ex1;
            //        int j = d % ex1;
            //        c[i, j] = d;
            //    });

            //    for (int d = 0; d < ex0 * ex1; ++d)
            //    {
            //        int i = d / ex1;
            //        int j = d % ex1;
            //        if (b[i, j] != c[i, j])
            //            throw new Exception();
            //    }
            //}
            //{
            //    int ex0 = 3;
            //    int ex1 = 5;
            //    int ex2 = 2;
            //    int[,,] b = new int[ex0, ex1, ex2];
            //    for (int d = 0; d < ex0 * ex1 * ex2; ++d)
            //    {
            //        // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
            //        int i = d / (ex1 * ex2);
            //        int r = d % (ex1 * ex2);
            //        int j = r / ex2;
            //        int k = r % ex2;
            //        b[i, j, k] = d;
            //    }

            //    int[,,] c = new int[ex0, ex1, ex2];
            //    Campy.Parallel.For(ex0 * ex1 * ex2, d =>
            //    {
            //        int i = d / (ex1 * ex2);
            //        int r = d % (ex1 * ex2);
            //        int j = r / ex2;
            //        int k = r % ex2;
            //        c[i, j, k] = d;
            //    });

            //    for (int d = 0; d < ex0 * ex1 * ex2; ++d)
            //    {
            //        // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
            //        int i = d / (ex1 * ex2);
            //        int r = d % (ex1 * ex2);
            //        int j = r / ex2;
            //        int k = r % ex2;
            //        if (b[i, j, k] != c[i, j, k])
            //            throw new Exception();
            //    }
            //}

            {
                int ex0 = 3;
                int ex1 = 5;
                // three rows, five columns.
                int[,] b = new int[ex0, ex1];
                for (int d = 0; d < ex0 * ex1; ++d)
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    b[i, j] = d;
                }

                int[,] c = new int[ex0, ex1];
                Campy.Parallel.For(15, d =>
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    c[i, j] = b[i, j];
                });

                for (int d = 0; d < ex0 * ex1; ++d)
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    if (b[i, j] != c[i, j])
                        throw new Exception();
                }
            }
        }
    }
}
