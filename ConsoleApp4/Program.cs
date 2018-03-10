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
    public class BitonicSorter
    {
        private static void Swap(ref int x, ref int y)
        {
            var t = x;
            x = y;
            y = t;
        }

        public static void BitonicSortParallel1(int[] a)
        {
            Parallel.Delay();
            uint N = (uint)a.Length;
            int log2n = Bithacks.FloorLog2(N);
            for (int k = 0; k < log2n; ++k)
            {
                uint n2 = N / 2;
                int twok = Bithacks.Power2(k);
                Campy.Parallel.For((int)n2, i =>
                {
                    int imp2 = i % twok;
                    int cross = imp2 + 2 * twok * (int)(i / twok);
                    int paired = -1 - imp2 + 2 * twok * (int)((i + twok) / twok);
                    if (a[cross] > a[paired]) Swap(ref a[cross], ref a[paired]);
                });
                for (int j = k - 1; j >= 0; --j)
                {
                    int twoj = Bithacks.Power2(j);
                    Campy.Parallel.For((int)n2, i =>
                    {
                        int imp2 = i % twoj;
                        int cross = imp2 + 2 * twoj * (int)(i / twoj);
                        int paired = cross + twoj;
                        if (a[cross] > a[paired]) Swap(ref a[cross], ref a[paired]);
                    });
                }
            }
            Parallel.Synch();
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

            var b = new BitonicSorter();
            Random rnd = new Random();
            int N = Bithacks.Power2(20);
            var a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            BitonicSorter.BitonicSortParallel1(a);
            for (int i = 0; i < N; ++i)
                if (a[i] != i)
                    throw new Exception();
        }
    }
}
