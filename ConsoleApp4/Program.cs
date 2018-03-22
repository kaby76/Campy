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
        public static void swap(ref int i, ref int j)
        {
            int t = i;
            i = j;
            j = t;
        }

        // [Bat 68]	K.E. Batcher: Sorting Networks and their Applications. Proc. AFIPS Spring Joint Comput. Conf., Vol. 32, 307-314 (1968)
        // Work inefficient sort, because half the threads are unused.
        public static void BitonicSort1(int[] a)
        {
            Parallel.Sticky(a);
            uint N = (uint)a.Length;
            int term = Bithacks.FloorLog2(N);
            for (int kk = 2; kk <= N; kk *= 2)
            {
                for (int jj = kk >> 1; jj > 0; jj = jj >> 1)
                {
                    int k = kk;
                    int j = jj;
                    Campy.Parallel.For((int)N, (i) =>
                    {
                        int ij = i ^ j;
                        if (ij > i)
                        {
                            if ((i & k) == 0)
                            {
                                if (a[i] > a[ij]) swap(ref a[i], ref a[ij]);
                            }
                            else // ((i & k) != 0)
                            {
                                if (a[i] < a[ij]) swap(ref a[i], ref a[ij]);
                            }
                        }
                    });
                }
            }
            Parallel.Sync();
        }

        public static void BitonicSort2(int[] a)
        {
            Parallel.Sticky(a);
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
                    if (a[cross] > a[paired])
                    {
                        int t = a[cross];
                        a[cross] = a[paired];
                        a[paired] = t;
                    }
                });
                for (int j = k - 1; j >= 0; --j)
                {
                    int twoj = Bithacks.Power2(j);
                    Campy.Parallel.For((int)n2, i =>
                    {
                        int imp2 = i % twoj;
                        int cross = imp2 + 2 * twoj * (int)(i / twoj);
                        int paired = cross + twoj;
                        if (a[cross] > a[paired])
                        {
                            int t = a[cross];
                            a[cross] = a[paired];
                            a[paired] = t;
                        }
                    });
                }
            }
            Parallel.Sync();
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
            Random rnd = new Random();

            int n = 4;
            int[][] jagged_array = new int[n][];

            Campy.Parallel.For(n,
                i =>
                {
                    jagged_array[i] = new int[i+1];
                });
        }
    }
}
