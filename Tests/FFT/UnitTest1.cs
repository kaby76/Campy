using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Numerics;

namespace FFTNS
{
    [TestClass]
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

        bool ApproxEqual(double a, double b)
        {
            if (b > a)
                return (b - a) < 0.01;
            else
                return (a - b) < 0.01;
        }

        [TestMethod]
        public void FFT_Test()
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
}
