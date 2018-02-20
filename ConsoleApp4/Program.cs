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
    static class Foo
    {
        public static IEnumerable<IEnumerable<T>> Split<T>(this T[] array, int size)
        {
            for (var i = 0; i < (float)array.Length / size; i++)
            {
                yield return array.Skip(i * size).Take(size);
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
        }

        public class BitonicSorter
        {
            private int[] a;
            // sorting direction:
            private static bool ASCENDING = true, DESCENDING = false;

            public void sort1(int[] a_)
            {
                a = a_;
                BitonicSort1();
            }

            private void bitonicSort(int lo, int n, bool dir)
            {
                if (n > 1)
                {
                    int m = n / 2;
                    bitonicSort(lo, m, ASCENDING);
                    bitonicSort(lo + m, m, DESCENDING);
                    bitonicMerge(lo, n, dir);
                }
            }

            private void bitonicMerge(int lo, int n, bool dir)
            {
                if (n > 1)
                {
                    int m = n / 2;
                    for (int i = lo; i < lo + m; i++)
                        compare(i, i + m, dir);
                    bitonicMerge(lo, m, dir);
                    bitonicMerge(lo + m, m, dir);
                }
            }

            private void compare(int i, int j, bool dir)
            {
                if (dir == (a[i] > a[j]))
                    swap(i, j);
            }

            private void swap(int i, int j)
            {
                int t = a[i];
                a[i] = a[j];
                a[j] = t;
            }

            // [Bat 68]	K.E. Batcher: Sorting Networks and their Applications. Proc. AFIPS Spring Joint Comput. Conf., Vol. 32, 307-314 (1968)

            void BitonicSort1()
            {
                //Parallel.Delay();
                uint N = (uint)a.Length;
                int term = Bithacks.FloorLog2(N);
                for (int k = 2; k <= N; k *= 2)
                {
                    for (int j = k >> 1; j > 0; j = j >> 1)
                    {
                        Campy.Parallel.For((int)N, (i) =>
                        {
                            int ij = i ^ j;
                            if (ij > i)
                            {
                                if ((i & k) == 0)
                                {
                                    if (a[i] > a[ij]) swap(i, ij);
                                }

                                if ((i & k) != 0)
                                {
                                    if (a[i] < a[ij]) swap(i, ij);
                                }
                            }
                        });
                        System.Console.Write("{0,3} {1,3} ", k, j);
                        for (int i = 0; i < N; ++i) System.Console.Write("{0,3}", a[i]);
                        System.Console.WriteLine();
                    }
                }
                //Parallel.Synch();
            }
        }

        static void Main(string[] args)
        {
            //StartDebugging();
            //Campy.Parallel.For((int)8, (i) =>
            //{
            //    int ij = i ^ j;
            //    if ((ij) > i)
            //    {
            //        if ((i & k) == 0 && a[i] > a[ij])
            //        {
            //            {
            //                int t = a[i];
            //                a[i] = a[ij];
            //                a[ij] = t;
            //            }
            //        }

            //        if ((i & k) != 0 && a[i] < a[ij])
            //        {
            //            {
            //                int t = a[i];
            //                a[i] = a[ij];
            //                a[ij] = t;
            //            }
            //        }
            //    }
            //});


            var b = new BitonicSorter();
            Random rnd = new Random();
            {
                int N = 16;
                b.sort1(Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray());
            }
            {
                int N = 16 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2;
                b.sort1(Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray());
            }
            //{
            //    b.sort1(Enumerable.Range(0, 8).ToArray().OrderBy(x => rnd.Next()).ToArray());
            //}
        }
    }
}
