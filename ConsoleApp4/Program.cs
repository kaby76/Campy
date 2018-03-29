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
    // https://stackoverflow.com/questions/463105/in-place-radix-sort
    // https://arxiv.org/abs/0706.4107
    // https://dl.acm.org/citation.cfm?id=1778601
    // https://www.geeksforgeeks.org/radix-sort/
    // http://www.geekviewpoint.com/java/sorting/radixsort
    // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.404.8825&rep=rep1&type=pdf
    // https://www.cs.princeton.edu/~rs/AlgsDS07/18RadixSort.pdf
    // https://rosettacode.org/wiki/Sorting_algorithms/Radix_sort#C.23

    class RadixSort
    {
        public static void Sort(int[] e)
        {
            int i, j;
            int[] tmp = new int[e.Length];
            for (int shift = 31; shift > -1; --shift)
            {
                j = 0;
                for (i = 0; i < e.Length; ++i)
                {
                    bool move = (e[i] << shift) >= 0;
                    if (shift == 0 ? !move : move)  // shift the 0's to old's head
                        e[i - j] = e[i];
                    else                            // move the 1's to tmp
                        tmp[j++] = e[i];
                }
                Array.Copy(tmp, 0, e, e.Length - j, j);
            }
        }

        public static void SortP(int[] e)
        {
            int[] tmp = new int[e.Length];
            for (int shift = 31; shift > -1; --shift)
            {
                int j = 0;
                Campy.Parallel.For(e.Length, i =>
                {
                    bool move = (e[i] << shift) >= 0;
                    if (shift == 0 ? !move : move) // shift the 0's to old's head
                        e[i - j] = e[i];
                    else // move the 1's to tmp
                        tmp[j++] = e[i]; // THIS IS UNSAFE CODE DUE TO J BEING INCREMENTED AMONG VARIOUS THREADS.
                });
                Array.Copy(tmp, 0, e, e.Length - j, j);
            }
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
            int N = 8;
            int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            RadixSort.SortP(a);
            for (int i = 0; i < N; ++i)
                if (a[i] != i)
                    throw new Exception();
        }
    }
}
