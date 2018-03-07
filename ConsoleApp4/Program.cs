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

            public void SortPar(int[] a_)
            {
                a = a_;
                BitonicSortPar();
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

            void BitonicSortSeq()
            {
                uint N = (uint)a.Length;
                int term = Bithacks.FloorLog2(N);
                for (int kk = 2; kk <= N; kk *= 2)
                {
                    for (int jj = kk >> 1; jj > 0; jj = jj >> 1)
                    {
                        int k = kk;
                        int j = jj;
                        Campy.Sequential.For((int)N, (i) =>
                        {
                            int ij = i ^ j;
                            if (ij > i)
                            {
                                if ((i & k) == 0)
                                {
                                    if (a[i] > a[ij]) swap(i, ij);
                                }
                                else // ((i & k) != 0)
                                {
                                    if (a[i] < a[ij]) swap(i, ij);
                                }
                            }
                        });
                    }
                }
            }

            void BitonicSortPar()
            {
                Campy.Parallel.Delay();
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
                                    if (a[i] > a[ij]) swap(i, ij);
                                }
                                else // ((i & k) != 0)
                                {
                                    if (a[i] < a[ij]) swap(i, ij);
                                }
                            }
                        });
                    }
                }
                Campy.Parallel.Synch();
            }
        }

        static void Main(string[] args)
        {
            StartDebugging();
            int n = 4;
            var t1 = new List<int>();
            for (int i = 0; i < n; ++i) t1.Add(0);
            Campy.Parallel.For(n, i =>
            {
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
    }
}
