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
            Campy.Parallel.Delay(a);
            int N = a.Length;
            int gap = N;
            bool swaps = true;
            float gap_factor = (float)1.25;

            while (gap > 1 || swaps)
            {
                int local_gap = (int)(gap / gap_factor);
                if (local_gap < 1) local_gap = 1;
                gap = local_gap;
                swaps = false;
                Campy.KernelType de = i =>
                {
                    if (a[i] > a[i + local_gap])
                    {
                        swap(ref a[i], ref a[i + local_gap]);
                        swaps = true;
                    }
                };
                if (gap != 1) Campy.Parallel.For(N - local_gap, de);
                else Campy.Sequential.For(N - gap, de);
            }
            Campy.Parallel.Synch();
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
            StartDebugging();
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
