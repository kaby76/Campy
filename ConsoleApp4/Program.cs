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

    public class UnitTest1
    {
        class EvenOddSorter
        {
            public static void swap(ref int i, ref int j)
            {
                int t = i;
                i = j;
                j = t;
            }

            public static void Sort(int[] a)
            {
                Campy.Parallel.Sticky(a);
                int N = a.Length;
                bool sorted = false;
                while (!sorted)
                {
                    sorted = true;
                    int n2 = N / 2;
                    Campy.Parallel.For(n2, i =>
                    {
                        int j = i * 2;
                        if (a[j] > a[j + 1])
                        {
                            swap(ref a[j], ref a[j + 1]);
                            sorted = false;
                        }
                    });
                    Campy.Parallel.For(n2 - 1, i =>
                    {
                        int j = i * 2 + 1;
                        if (a[j] > a[j + 1])
                        {
                            swap(ref a[j], ref a[j + 1]);
                            sorted = false;
                        }
                    });
                }
                Campy.Parallel.Sync();
            }

            public static void SeqSort(int[] a)
            {
                int N = a.Length;
                bool sorted = false;
                while (!sorted)
                {
                    sorted = true;
                    int n2 = N / 2;
                    for (int i = 0; i < n2; ++i)
                    {
                        int j = i * 2;
                        if (a[j] > a[j + 1])
                        {
                            swap(ref a[j], ref a[j + 1]);
                            sorted = false;
                        }
                    }
                    for (int i = 0; i < n2 - 1; ++i)
                    {
                        int j = i * 2 + 1;
                        if (a[j] > a[j + 1])
                        {
                            swap(ref a[j], ref a[j + 1]);
                            sorted = false;
                        }
                    }
                }
            }
        }

        public static void Test1()
        {
            Random rnd = new Random();
            int N = 8;
            int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            EvenOddSorter.Sort(a);
            for (int i = 0; i < N; ++i)
                if (a[i] != i)
                    throw new Exception();
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
            UnitTest1.Test1();
        }
    }
}
