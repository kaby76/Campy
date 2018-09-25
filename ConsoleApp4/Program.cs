using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using Campy;
using System.Linq;

namespace ConsoleApp4
{
    class OddEvenSort
    {
        public static void swap(ref int i, ref int j)
        {
            int t = i;
            i = j;
            j = t;
        }

        public static void Seq(int[] a)
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

        public static void Par(int[] a)
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

        // Adapted from http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm

        /** sorts a piece of length n of the array
         * starting at position lo
         */
        public static void Rec(int[] a, int lo, int n)
        {
            if (n > 1)
            {
                int m = n / 2;
                Rec(a, lo, m);
                Rec(a, lo + m, m);
                RecMerge(a, lo, n, 1);
            }
        }

        /** lo is the starting position and
         * n is the length of the piece to be merged,
         * r is the distance of the elements to be compared
         */
        public static void RecMerge(int[] a, int lo, int n, int r)
        {
            int m = r * 2;
            if (m < n)
            {
                RecMerge(a, lo, n, m); // even subsequence
                RecMerge(a, lo + r, n, m); // odd subsequence
                for (int i = lo + r; i + r < lo + n; i += m)
                    if (a[i] > a[i + r])
                        swap(ref a[i], ref a[i + r]);
            }
            else if (a[lo] > a[lo + r])
                swap(ref a[lo], ref a[lo + r]);
        }
    }

    public class UnitTest1
    {
        public static void Test1()
        {
            Random rnd = new Random();
            int N = 8;
            {
                int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
                OddEvenSort.Seq(a);
                for (int i = 0; i < N; ++i)
                    if (a[i] != i)
                        throw new Exception();
            }
            {
                int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
                OddEvenSort.Par(a);
                for (int i = 0; i < N; ++i)
                    if (a[i] != i)
                        throw new Exception();
            }
        }
    }

    class Program
    {
        static void StartDebugging()
        {
            //Campy.Utils.Options.Set("debug_info_off");
            //Campy.Utils.Options.Set("graph_trace");
            //Campy.Utils.Options.Set("name_trace");
            //     Campy.Utils.Options.Set("cfg_construction_trace");
 //                 Campy.Utils.Options.Set("cfg_construction_trace");
            //Campy.Utils.Options.Set("dot_graph");
            //Campy.Utils.Options.Set("jit_trace");
            //Campy.Utils.Options.Set("memory_trace");
            //Campy.Utils.Options.Set("ptx_trace");
            //Campy.Utils.Options.Set("state_computation_trace");
            Campy.Utils.Options.Set("overview_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_llvm_computation_trace");
            // Campy.Utils.Options.Set("continue_with_no_resolve");
            //Campy.Utils.Options.Set("copy_trace");
            //    Campy.Utils.Options.Set("runtime_trace");
            Campy.Utils.Options.Set("graph-output");
            Campy.Utils.Options.Set("ptx-output");
            Campy.Utils.Options.Set("llvm-output");
            Campy.Utils.Options.Set("dot-output");
            //Campy.Utils.Options.Set("import-only");
            Campy.Utils.Options.Set("trace-cctors");
        }

        static void Main(string[] args)
        {
            StartDebugging();
            UnitTest1.Test1();
        }
    }
}
