using System;
using System.Linq;
using System.Numerics;

namespace ConsoleApp4
{
    class A
    {
        public int X { get; set; }

        public int Score(A b)
        {
            return X + b.X;
        }
    }

    public class BatcherOddEvenMergeSort
    {
        public static void swap(ref int i, ref int j)
        {
            int t = i;
            i = j;
            j = t;
        }

        // Adapted from http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm

        /** sorts a piece of length n of the array
         * starting at position lo
         */
        public static void OddEvenMergeSort(int[] a, int lo, int n)
        {
            if (n > 1)
            {
                int m = n / 2;
                OddEvenMergeSort(a, lo, m);
                OddEvenMergeSort(a, lo + m, m);
                OddEvenMerge(a, lo, n, 1);
            }
        }

        /** lo is the starting position and
         * n is the length of the piece to be merged,
         * r is the distance of the elements to be compared
         */
        public static void OddEvenMerge(int[] a, int lo, int n, int r)
        {
            int m = r * 2;
            if (m < n)
            {
                OddEvenMerge(a, lo, n, m); // even subsequence
                OddEvenMerge(a, lo + r, n, m); // odd subsequence
                for (int i = lo + r; i + r < lo + n; i += m)
                {
                    System.Console.WriteLine(i + " cmp " + (i + r));
                    if (a[i] > a[i + r])
                        swap(ref a[i], ref a[i + r]);
                }
            }
            else
            {
                if (a[lo] > a[lo + r])
                    swap(ref a[lo], ref a[lo + r]);
                System.Console.WriteLine(lo + " cmp " + (lo + r));
            }
        }

        // https://stackoverflow.com/questions/34426337/how-to-fix-this-non-recursive-odd-even-merge-sort-algorithm
        public static void show_pairs(int n)
        {
            for (int p = 1; p < n; p += p)
            {
                System.Console.WriteLine("p " + p);
                for (int k = p; k > 0; k /= 2)
                {
                    System.Console.WriteLine("k " + k);
                    int total_j = k;
                    //System.Console.WriteLine("total j " + total_j);
                    int times = n - k - k % p;
                    int times1 = n + 1 - k - k % p;
                    //System.Console.WriteLine("times " + times);
                    //System.Console.WriteLine("times1 " + times1);
                    int total_i = (n - k - k % p) / (2 * k);
                    int total_i1 = (n + 1 - k - k % p) / (2 * k);
                    //System.Console.WriteLine("total i " + total_i);
                    //System.Console.WriteLine("total i1 " + total_i1);
                    // "k" indicates how many "groups"
                    // "p" indicates distance.
                    // First group starts at index of k % p.
                    for (int j = 0; j < k; j++)
                    {
                        //System.Console.WriteLine("j " + j);
                        for (int i = k % p; i + k < n; i += k + k)
                        {
                            //System.Console.WriteLine("i " + i);
                            if ((i + j) / (p + p) == (i + j + k) / (p + p))
                                System.Console.WriteLine((i + j).ToString() + " cmp " + (i + j + k).ToString());
                        }
                    }
                    System.Console.WriteLine();
                }
            }
        }
        // https://codereview.stackexchange.com/questions/114824/constructing-an-odd-even-mergesort-sorting-network
        public static void show_pairs2(int n)
        {
            int length = n;
            int groups = Bithacks.CeilingLog2((uint)length);
            for (int group = 0; group < groups; group++)
            {
                System.Console.WriteLine("group " + group);
                int blocks = 1 << (groups - group - 1);
                for (int block = 0; block < blocks; block++)
                {
                    System.Console.WriteLine("block " + block);
                    for (int stage = 0; stage <= group; stage++)
                    {
                        int distance = 1 << (group - stage);
                        int startPoint = (stage == 0) ? 0 : distance;
                        for (int j = startPoint; j + distance < (2 << group); j += 2 * distance)
                        {
                            for (int i = 0; i < distance; i++)            // shift startpoints
                            {
                                int x = (block * (length / blocks)) + j + i;
                                int y = x + distance;
                                System.Console.WriteLine(x + " cmp " + y);
                            }
                        }
                    }
                }
                System.Console.WriteLine();
            }
        }
    }

    public class CheckSortingNetwork
    {
        public static void swap(ref int i, ref int j)
        {
            int t = i;
            i = j;
            j = t;
        }

        public static void Sort1(int[] e)
        {
            // Implementation of diagram at https://en.wikipedia.org/wiki/Sorting_network
            // Does this really sort?
            if (e[0] > e[2]) swap(ref e[0], ref e[2]);
            if (e[1] > e[3]) swap(ref e[1], ref e[3]);

            if (e[0] > e[1]) swap(ref e[0], ref e[1]);
            if (e[2] > e[3]) swap(ref e[2], ref e[3]);

            if (e[1] > e[2]) swap(ref e[1], ref e[2]);
        }

        public static void Sort2(int[] e)
        {
            if (e[0] > e[1]) swap(ref e[0], ref e[1]);
            if (e[2] > e[3]) swap(ref e[2], ref e[3]);

            if (e[0] > e[2]) swap(ref e[0], ref e[2]);
            if (e[1] > e[3]) swap(ref e[1], ref e[3]);

            if (e[1] > e[2]) swap(ref e[1], ref e[2]);
        }

        public static void Sort3(int[] e)
        {
            if (e[1] > e[2]) swap(ref e[1], ref e[2]);

            if (e[0] > e[1]) swap(ref e[0], ref e[1]);
            if (e[2] > e[3]) swap(ref e[2], ref e[3]);

            if (e[0] > e[2]) swap(ref e[0], ref e[2]);
            if (e[1] > e[3]) swap(ref e[1], ref e[3]);
        }
    }

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

    public class FFT
    {
        /* Performs a Bit Reversal Algorithm on a postive integer 
         * for given number of bits
         * e.g. 011 with 3 bits is reversed to 110
         */
        public static int BitReverse(int n, int bits)
        {
            int reversedN = n;
            int count = bits - 1;

            n >>= 1;
            while (n > 0)
            {
                reversedN = (reversedN << 1) | (n & 1);
                count--;
                n >>= 1;
            }

            return ((reversedN << count) & ((1 << bits) - 1));
        }

        /* Uses Cooley-Tukey iterative in-place algorithm with radix-2 DIT case
         * assumes no of points provided are a power of 2 */
        public static void Seq(Complex[] buffer)
        {
            int bits = (int)Math.Log(buffer.Length, 2);
            for (int j = 1; j < buffer.Length / 2; j++)
            {
                int swapPos = BitReverse(j, bits);
                var temp = buffer[j];
                buffer[j] = buffer[swapPos];
                buffer[swapPos] = temp;
            }

            for (int N = 2; N <= buffer.Length; N <<= 1)
            {
                for (int i = 0; i < buffer.Length; i += N)
                {
                    for (int k = 0; k < N / 2; k++)
                    {
                        int evenIndex = i + k;
                        int oddIndex = i + k + (N / 2);
                        var even = buffer[evenIndex];
                        var odd = buffer[oddIndex];

                        double term = -2 * Math.PI * k / (double)N;
                        Complex exp = new Complex(Math.Cos(term), Math.Sin(term)) * odd;

                        buffer[evenIndex] = even + exp;
                        buffer[oddIndex] = even - exp;
                    }
                }
            }
        }

        public static void Par(Complex[] buffer)
        {
            Campy.Parallel.Sticky(buffer);

            int bits = (int)Math.Log(buffer.Length, 2);
            Campy.Parallel.For(buffer.Length / 2 - 1, k =>
            {
                int j = k + 1;
                System.Console.WriteLine(k + " " + j);
                int swapPos = BitReverse(j, bits);
                var temp = buffer[j];
                buffer[j] = buffer[swapPos];
                buffer[swapPos] = temp;
            });

            for (int N = 2; N <= buffer.Length; N <<= 1)
            {
                int step = N / 2;
                Campy.Parallel.For(buffer.Length / 2, d =>
                {
                    var k = d % step;
                    var t = d % step + N * (d / step);
                    int evenIndex = t;
                    int oddIndex = t + step;

                    var even = buffer[evenIndex];
                    var odd = buffer[oddIndex];

                    double term = -2 * Math.PI * k / (double)N;
                    Complex exp = new Complex(Math.Cos(term), Math.Sin(term)) * odd;

                    buffer[evenIndex] = even + exp;
                    buffer[oddIndex] = even - exp;
                });
            }
            Campy.Parallel.Sync();
        }

        static bool ApproxEqual(double a, double b)
        {
            return b - a < 0.0001 || a - b < 0.0001;
        }

        public static void FFT_Test()
        {
            Complex[] input = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            var copy = input.ToArray();

            Par(input);
            Seq(copy);

            for (int i = 0; i < input.Length; ++i)
            {
                if (!ApproxEqual(copy[i].Real, input[i].Real)) throw new Exception();
                if (!ApproxEqual(copy[i].Imaginary, input[i].Imaginary)) throw new Exception();
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
            //{
            //    Random rnd = new Random();
            //    for (; ; )
            //    {
            //        int N = 4;
            //        int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            //        System.Console.WriteLine(String.Join(" ", a));
            //        CheckSortingNetwork.Sort3(a);
            //        System.Console.WriteLine(String.Join(" ", a));
            //        for (int i = 0; i < N; ++i)
            //            if (a[i] != i)
            //                throw new Exception();
            //        System.Console.WriteLine();
            //    }
            //}

            //StartDebugging();

            {
                A[] array = new A[10];
                for (int i = 0; i < 10; ++i) array[i] = new A();

                Campy.Parallel.For(10, i =>
                {
                    array[i].X = i;
                });

                for (int i = 0; i < 10; i++)
                {
                    if (array[i].X != i) throw new Exception();
                }
            }

            //string[] strings = new string[] {"a", "bb", "ccc"};
            //int[] len = new int[strings.Length];
            //Campy.Parallel.For(strings.Length, i =>
            //{
            //    len[i] = strings[i][0];
            //});

            //{
            //    Random rnd = new Random();
            //    int N = 32;
            //    int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            //    BatcherOddEvenMergeSort.OddEvenMergeSort(a, 0, N);
            //    for (int i = 0; i < N; ++i)
            //        if (a[i] != i)
            //            throw new Exception();
            //    System.Console.WriteLine("---------------------------");
            //    BatcherOddEvenMergeSort.show_pairs(N);
            //    System.Console.WriteLine("---------------------------");
            //    BatcherOddEvenMergeSort.show_pairs2(N);
            //}
            {
                Random rnd = new Random();
                int N = 8;
                int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
                EvenOddSorter.Sort(a);
                for (int i = 0; i < N; ++i)
                    if (a[i] != i)
                        throw new Exception();
            }

            //FFT.FFT_Test();
        }
    }
}
