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

    public class FFTC
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
        public static void FFT(Complex[] buffer)
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

        public static void FFTGPU(Complex[] buffer)
        {
            int bits = (int)Math.Log(buffer.Length, 2);

            Campy.Parallel.For(buffer.Length / 2 - 1, k =>
            {
                int j = k + 1;
                int swapPos = BitReverse(j, bits);
                var temp = buffer[j];
                buffer[j] = buffer[swapPos];
                buffer[swapPos] = temp;
            });

            for (int N = 2; N <= buffer.Length; N <<= 1)
            {
                int step = N / 2;
                int bstep = N;
                Campy.Parallel.For(buffer.Length / 2, d =>
                {
                    var k = d % step;
                    var i = N * (d / step);
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
        }

        static bool ApproxEqual(double a, double b)
        {
            if (b > a)
                return (b - a) < 0.01;
            else
                return (a - b) < 0.01;
        }

        public static void FFT_Test()
        {
            Complex[] input = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            var copy = input.ToArray();

            FFTGPU(input);
            FFT(copy);

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
            StartDebugging();
            FFTC.FFT_Test();
            UnitTest1.Test1();
        }
    }
}
