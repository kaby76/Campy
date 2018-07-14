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
            //Campy.Parallel.For(3, i =>
            //{
            //    //System.Console.WriteLine(i); // no explicit conversion.
            //    System.Console.WriteLine(i.ToString()); // value converted explicitly in code.
            //});
            // List of ints.
            List<int> x = new List<int>();
            int n = 4;
            for (int i = 0; i < n; ++i) x.Add(0);
            Campy.Parallel.For(n, i =>
            {
                x[i] = i;
            });
            for (int i = 0; i < n; ++i) if (x[i] != i)
                    throw new Exception();
            //bool c1;
            //bool c2;
            //bool c3;
            //bool c4;
            //bool c5;
            //bool c6;
            //bool c7;
            //bool c8;
            //int a = 1;
            //int b = 1;
            //int c = 2;
            //Campy.Parallel.For(1, i =>
            //{
            //    c1 = a > c;
            //    c2 = a < c;
            //    c3 = a >= c;
            //    c4 = a <= c;
            //    c5 = a == c;
            //    c6 = a == b;
            //    c7 = a != b;
            //    c8 = a != c;
            //});
            //int c1;
            //int c2;
            //int c3;
            //int c4;
            //int c5;
            //int c6;
            //int c7;
            //int c8;
            //int a = 1;
            //int b = 1;
            //int c = 2;
            //Campy.Parallel.For(1, i =>
            //{
            //    c1 = a > c ? 0 : 2;
            //    c2 = a < c ? 0 : 2;
            //    c3 = a >= c ? 0 : 2;
            //    c4 = a <= c ? 0 : 2;
            //    c5 = a == c ? 0 : 2;
            //    c6 = a == b ? 0 : 2;
            //    c7 = a != b ? 0 : 2;
            //    c8 = a != c ? 0 : 2;
            //});
        }
    }
}
