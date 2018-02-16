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

            public void sort(int[] a_)
            {
                a = a_;
                bitonicSort(0, a.Length, ASCENDING);
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
                    exchange(i, j);
            }

            private void exchange(int i, int j)
            {
                int t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }

        public class Reduction
        {
            public static void ReductionUsingArrays()
            {
                int n = Bithacks.Power2(10);
                int result_gpu = 0;
                int result_cpu = 0;
                {
                    Parallel.Delay();
                    int[] data = new int[n];
                    Campy.Parallel.For(n, idx => data[idx] = 1);
                    for (int level = 1; level <= Bithacks.Log2(n); level++)
                    {
                        int step = Bithacks.Power2(level);
                        Campy.Parallel.For(new Extent(n / step), idx =>
                        {
                            var i = step * idx;
                            data[i] = data[i] + data[i + step / 2];
                        });
                    }
                    Parallel.Synch();
                    result_gpu = data[0];
                }
                {
                    int[] data = new int[n];
                    for (int idx = 0; idx < n; ++idx) data[idx] = 1;
                    for (int level = 1; level <= Bithacks.Log2(n); level++)
                    {
                        int step = Bithacks.Power2(level);
                        for (int idx = 0; idx < n / step; idx++)
                        {
                            var i = step * idx;
                            data[i] = data[i] + data[i + step / 2];
                        }
                    }
                    result_cpu = data[0];
                }
                if (result_gpu != result_cpu) throw new Exception();
            }

            public static void ReductionUsingLists()
            {
                Parallel.Delay();
                int n = Bithacks.Power2(10);
                float result_gpu = 0;
                float result_cpu = 0;
                {
                    List<float> data = Enumerable.Range(0, n).Select(i => ((float)i) / 10).ToList();
                    for (int level = 1; level <= Bithacks.Log2(n); level++)
                    {
                        int step = Bithacks.Power2(level);
                        for (int idx = 0; idx < n / step; idx++)
                        {
                            var i = step * idx;
                            data[i] = data[i] + data[i + step / 2];
                        }
                    }
                    result_cpu = data[0];
                }
                {
                    List<float> data = Enumerable.Range(0, n).Select(i => ((float)i) / 10).ToList();
                    for (int level = 1; level <= Bithacks.Log2(n); level++)
                    {
                        int step = Bithacks.Power2(level);
                        Campy.Parallel.For(new Extent(n / step), idx =>
                        {
                            var i = step * idx;
                            data[i] = data[i] + data[i + step / 2];
                        });
                    }
                    result_gpu = data[0];
                }
                Parallel.Synch();
                if (result_gpu != result_cpu) throw new Exception();
            }
        }

        static void Main(string[] args)
        {
            StartDebugging();
            Reduction.ReductionUsingArrays();
        }
    }
}
