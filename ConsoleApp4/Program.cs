using System;
using System.Collections.Generic;
using Campy;
using System.Numerics;

namespace ConsoleApp4
{
    class N
    {
        public int id;
        public int level;
        public bool visit;
        public bool visited;
        public N Left;
        public N Right;
    }

    class Program
    {
        private static int counter;

        static void MakeIt(int current_height, N current_node, ref List<N> all_nodes)
        {
            if (current_height == 0)
                return;
            current_height--;
            N l = new N();
            l.id = counter++;
            all_nodes.Add(l);

            N r = new N();
            r.id = counter++;
            all_nodes.Add(r);

            current_node.Left = l;
            current_node.Right = r;

            MakeIt(current_height, current_node.Left, ref all_nodes);
            MakeIt(current_height, current_node.Right, ref all_nodes);
        }
        /* Performs a Bit Reversal Algorithm on a postive integer 
    * for given number of bits
    * e.g. 011 with 3 bits is reversed to 110 */
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
         * assumes no of points provided are a power of 2 */
        public static void FFTGPU(Complex[] buffer)
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


        static void Main(string[] args)
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
            //{
            //    int[][] jagged_array = new int[][]
            //    {
            //        new int[] {1, 3, 5, 7, 9},
            //        new int[] {0, 2, 4, 6},
            //        new int[] {11, 22}
            //    };
            //    Campy.Parallel.For(3, i =>
            //    {
            //        jagged_array[i][0] = i; //jagged_array[i].Length;
            //    });
            //    for (int i = 0; i < 3; ++i)
            //        if (jagged_array[i][0] != i) // jagged_array[i].Length)
            //            throw new Exception("unequal");
            //    Campy.Parallel.For(3, i =>
            //    {
            //        jagged_array[i][0] = jagged_array[i].Length;
            //    });
            //    for (int i = 0; i < 3; ++i)
            //        if (jagged_array[i][0] != jagged_array[i].Length)
            //            throw new Exception("unequal");
            //}

            {
                // List of ints.
                List<List<int>> x = new List<List<int>>();
                int n = 4;
                for (int i = 0; i < n; ++i)
                    x.Add(new List<int>());
                for (int i = 0; i < n; ++i)
                    x[i].Add(0);
                Campy.Parallel.For(n, i => x[i][0] = i);
                for (int i = 0; i < n; ++i)
                    if (x[i][0] != i)
                        throw new Exception("unequal");
            }


            {
                int e = 10;
                Extent ex = new Extent(3, 5); // three rows, five columns.
                int[,] b = new int[ex[0], ex[1]];
                for (int i = 0; i < ex[0]; ++i)
                    for (int j = 0; j < ex[1]; ++j)
                        b[i, j] = (i + 1) * (j + 1);
                Campy.Parallel.For(ex, d =>
                {
                    int i = d[0];
                    int j = d[0];
                    b[i, j] = b[i, j] * 2;
                });
            }
            {
                Complex[] input = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };

                FFT(input);
                FFTGPU(input);

                Console.WriteLine("Results:");
                foreach (Complex c in input)
                {
                    Console.WriteLine(c);
                }
            }


            {
                // Saxpy (vector update).
                int n = 2;
                double[] x = new double[n];
                double[] y = new double[n];
                float a = 10f;

                Campy.Parallel.For(n, i => x[i] = i);
                Campy.Parallel.For(n, i => y[i] = i - 1);
                Campy.Parallel.For(n, i =>
                {
                    y[i] = y[i] + a * x[i];
                });
                for (int i = 0; i < n; ++i)
                    System.Console.Write(y[i] + " ");
                System.Console.WriteLine();
            }

            //{
            //    int n = 4;
            //    int[] x = new int[n];
            //    Campy.Parallel.For(n, i => x[i[0]] = i[0]);
            //}

            //if (false)
            //{
            //    int max_level = 16;
            //    int n = Bithacks.Power2(max_level);
            //    List<int> data = Enumerable.Repeat(0, n).ToList();

            //    Campy.Parallel.For(n, idx => data[idx] = 1);
            //    for (int level = 1; level <= Bithacks.Log2(n); level++)
            //    {
            //        int step = Bithacks.Power2(level);
            //        Campy.Parallel.For(n / step, idx =>
            //        {
            //            var i = step * idx;
            //            data[i] = data[i] + data[i + step / 2];
            //        });
            //    }
            //    System.Console.WriteLine("sum = " + data[0]);
            //}

            //if (false)
            //{
            //    int max_level = 16;
            //    int n = Bithacks.Power2(max_level);
            //    int[] data = new int[n];

            //    Campy.Parallel.For(n, idx => data[idx] = 1);
            //    for (int level = 1; level <= Bithacks.Log2(n); level++)
            //    {
            //        int step = Bithacks.Power2(level);
            //        Campy.Parallel.For(n / step, idx =>
            //        {
            //            var i = step * idx;
            //            data[i] = data[i] + data[i + step / 2];
            //        });
            //    }
            //    System.Console.WriteLine("sum = " + data[0]);
            //}

            //if (false)
            //{
            //    int[][] jagged_array = new int[][]
            //    {
            //        new int[] {1,3,5,7,9},
            //        new int[] {0,2,4,6},
            //        new int[] {11,22}
            //    };

            //    Campy.Parallel.For(3, i =>
            //    {
            //        //int sum = 0;
            //        //for (int j = 0; j < jagged_array[i].Length; ++j)
            //        //{
            //        //    sum += jagged_array[i][j];
            //        //}
            //        jagged_array[i][0] = jagged_array[i].Length;
            //    });

            //}

            //if (false)
            //{
            //    // Create complete binary tree.
            //    int max_level = 6;
            //    N root = new N();
            //    counter++;
            //    List<N> all_nodes = new List<N>();
            //    all_nodes.Add(root);
            //    MakeIt(max_level, root, ref all_nodes);
            //    root.visit = true;
            //    int size = all_nodes.Count;
            //    for (; ;)
            //    {
            //        bool changed = false;
            //        for (int i = 0; i < size; ++i)
            //        {
            //            if (i >= size)
            //                continue;
            //            N node = all_nodes[i];
            //            if (!node.visit)
            //                continue;
            //            node.visit = false;
            //            node.visited = true;
            //            N l = node.Left;
            //            N r = node.Right;
            //            if (l != null)
            //            {
            //                l.visit = true;
            //                l.level = node.level + 1;
            //                changed = true;
            //            }
            //            if (r != null)
            //            {
            //                r.visit = true;
            //                r.level = node.level + 1;
            //                changed = true;
            //            }
            //        }
            //        if (!changed)
            //            break;
            //    }
            //    //Campy.Parallel.For(new Extent(size), i
            //    //    =>
            //    //{
            //    //    if (i >= size) return;
            //    //    N node = all_nodes[i];
            //    //    if (!node.visit)
            //    //        return;
            //    //    node.visit = false;
            //    //    node.visited = true;
            //    //    N l = node.Left;
            //    //    N r = node.Right;
            //    //    if (l != null)
            //    //    {
            //    //        l.visit = true;
            //    //        l.level = node.level + 1;
            //    //    }
            //    //    if (r != null)
            //    //    {
            //    //        r.visit = true;
            //    //        r.level = node.level + 1;
            //    //    }
            //    //});

            //    //for (int level = 0; level < max_level; ++level)
            //    //{
            //    //    Campy.Parallel.For(new Extent(Bithacks.Power2(level)), i =>
            //    //    {

            //    //    });
            //    //}
            //}
        }
    }
}
