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
            int[][] jagged_array = new int[][]
            {
                new int[] {1, 3, 5, 7, 9},
                new int[] {0, 2, 4, 6},
                new int[] {11, 22}
            };
            Campy.Parallel.For(3, i =>
            {
                jagged_array[i][0] = i; //jagged_array[i].Length;
            });

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
                int N = Bithacks.Power2(16);
                b.SortPar(Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray());
            }
        }
    }
}
