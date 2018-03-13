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
    class CombSorter
    {
        public static void swap(ref int i, ref int j)
        {
            int t = i;
            i = j;
            j = t;
        }

        public static void Sort(int[] a)
        {
            //Campy.Parallel.Delay();
            int N = a.Length;
            int global_gap = N;
            int swaps = 1;
            float gap_factor = (float)1.25;
            while (global_gap > 1 || swaps > 0)
            {
                System.Console.WriteLine(String.Join(" ", a));
                int local_gap = (int)(global_gap / gap_factor);
                if (local_gap < 1) local_gap = 1;
                System.Console.WriteLine(local_gap);
                System.Console.WriteLine(swaps);
                global_gap = local_gap;
                swaps = 0;
                KernelType k = i =>
                {
                    if (a[i] > a[i + local_gap])
                    {
                        {
                            int t = a[i];
                            a[i] = a[i + local_gap];
                            a[i + local_gap] = t;
                        }
                        swaps = 1;
                    }
                };
                if (local_gap != 1)
                    Campy.Parallel.For(N - local_gap, k);
                else
                    Campy.Sequential.For(N - local_gap, k);
                System.Console.WriteLine(String.Join(" ", a));
            }
            //Campy.Parallel.Synch();
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
        }
        static void Main(string[] args)
        {
            //StartDebugging();
            Random rnd = new Random();
            int N = Bithacks.Power2(4);
            var a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            CombSorter.Sort(a);
            for (int i = 0; i < N; ++i)
                if (a[i] != i)
                    throw new Exception();
        }
    }
}
