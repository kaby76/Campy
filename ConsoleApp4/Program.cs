using System;
using System.Collections.Generic;
using System.Linq;
using Campy;
using System.Numerics;

namespace ConsoleApp4
{
    class Program
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


        /* Uses Cooley-Tukey iterative in-place algorithm with radix-2 DIT case
         * assumes no of points provided are a power of 2
         */
        public static void FFTGPU(Complex[] buffer)
        {
            int bits = (int)Math.Log(buffer.Length, 2);

            var copy = buffer.ToArray();
            for (int j = 1; j < buffer.Length / 2; j++)
            {
                int swapPos = BitReverse(j, bits);
                var temp = copy[j];
                copy[j] = copy[swapPos];
                copy[swapPos] = temp;
            }


            Campy.Parallel.For(1, buffer.Length / 2, j =>
            {
                int swapPos = BitReverse(j, bits);
                var temp = buffer[j];
                buffer[j] = buffer[swapPos];
                buffer[swapPos] = temp;
            });

            for (int N = 2; N <= buffer.Length; N <<= 1)
            {
                int step = N / 2;
                int bstep = N;
                for (int i = 0; i < buffer.Length; i += N)
                {
                    for (int k = 0; k < N / 2; k++)
                    {
                        int evenIndex = i + k;
                        int oddIndex = i + k + (N / 2);

                        var even = copy[evenIndex];
                        var odd = copy[oddIndex];

                        double term = -2 * Math.PI * k / (double) N;
                        Complex exp = new Complex(Math.Cos(term), Math.Sin(term)) * odd;

                        copy[evenIndex] = even + exp;
                        copy[oddIndex] = even - exp;
                    }
                }

            }
            for (int N = 2; N <= buffer.Length; N <<= 1)
            {
                int step = N / 2;
                int bstep = N;
                Campy.Parallel.For(0, buffer.Length / 2, d =>
                {
                    var i = d % step + N * (d / step);
                    int evenIndex = i;
                    int oddIndex = i + step;
                    var even = buffer[evenIndex];
                    var odd = buffer[oddIndex];

                    double term = -2 * Math.PI * 1 / (double)N;
                    Complex exp = new Complex(Math.Cos(term), Math.Sin(term)) * odd;

                    buffer[evenIndex] = even + exp;
                    buffer[oddIndex] = even - exp;
                });
            }
        }

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

        static void Main(string[] args)
        {
            StartDebugging();
            //{
            //    int n = 4;

            //    var t1 = new List<int>();
            //    for (int i = 0; i < n; ++i) t1.Add(0);
            //    Campy.Parallel.For(n, i =>
            //    {
            //        if (i % 2 == 0)
            //            t1[i] = i * 20;
            //        else
            //            t1[i] = i * 30;
            //    });
            //    for (int i = 0; i < n; ++i)
            //        if (i % 2 == 0)
            //        {
            //            if (t1[i] != i * 20) throw new Exception();
            //        }
            //        else
            //        {
            //            if (t1[i] != i * 30) throw new Exception();
            //        }
            //}
            double pi = 3.141592653589793;
            int num = 10;
            double[] dodo1 = new double[num];
            double[] dodo2 = new double[num];
            double[] dodo3 = new double[num];


            //for (int i = 0; i < num; ++i)
            //{
            //    dodo1[i] = Campy.Compiler.Runtime.Sine(-0.1 * i);
            //}
            //Campy.Parallel.For(0, num, i =>
            //{
            //    dodo2[i] = Campy.Compiler.Runtime.Sine(-0.1 * i);
            //});
            //Campy.Parallel.For(0, num, i =>
            //{
            //    dodo3[i] = Math.Sin(-0.1 * i);
            //});

            Complex[] wonder1 = new Complex[num * 4];
            Complex[] wonder2 = new Complex[num * 4];
            Complex[] wonder3 = new Complex[num * 4];
            //Campy.Parallel.For(0, num, i =>
            //{
            //    double v = (4 * i) / 10;
            //    wonder1[i] = new Complex(1, 1);
            //});
            Campy.Parallel.For(0, num, i =>
            {
                double v = (4 * i) / 10;
                wonder2[i] = new Complex(1 + v, 1 + v);
            });
            Campy.Parallel.For(0, num, i =>
            {
                double v = 4.0 * i / 10;
                //Complex aa = new Complex(0.5 + v, 1 + v);
                //Complex bb = new Complex(1 + v, 1 + v);
                //Complex o1 = aa + bb;
                //wonder3[4 * i + 0] = o1;
                wonder3[4 * i + 0] = new Complex(0.5 + v, 1 + v) + new Complex(1 + v, 1 + v);
            });

            Campy.Parallel.For(0, num, i =>
            {
                double v = 4.0 * i / 10;
                Complex aa = new Complex(0.5 + v, 1 + v);
                Complex bb = new Complex(1 + v, 1 + v);
                Complex o1 = aa + bb;
                wonder3[4 * i + 0] = o1;
            });


            {
                Complex[] input = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

                //FFT(input);
                FFTGPU(input);

                Console.WriteLine("Results:");
                foreach (Complex c in input)
                {
                    Console.WriteLine(c);
                }
            }
        }
    }
}
