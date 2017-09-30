﻿using System;
using System.Collections.Generic;
using Campy;

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

        static void Main(string[] args)
        {
            Campy.Utils.Options.Set("graph_trace", true);
            Campy.Utils.Options.Set("module_trace", true);
            Campy.Utils.Options.Set("name_trace", true);
            Campy.Utils.Options.Set("cfg_construction_trace", true);
            Campy.Utils.Options.Set("dot_graph", true);
            Campy.Utils.Options.Set("jit_trace", true);
            Campy.Utils.Options.Set("memory_trace", true);
            Campy.Utils.Options.Set("ptx_trace", true);
            Campy.Utils.Options.Set("state_computation_trace", true);

            {
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
                        if (t1[i] != i * 20) throw new Exception("unequal");
                    }
                    else
                    {
                        if (t1[i] != i * 30) throw new Exception("unequal");
                    }

                var t2 = new List<float>();
                for (int i = 0; i < n; ++i) t2.Add(0);
                Campy.Parallel.For(n, i => t2[i] = 0.1f * i);
                for (int i = 0; i < n; ++i) if (t2[i] != 0.1f * i) throw new Exception("unequal");

                var t3 = new List<double>();
                for (int i = 0; i < n; ++i) t3.Add(0);
                Campy.Parallel.For(n, i => t3[i] = 0.1d * i);
                for (int i = 0; i < n; ++i) if (t3[i] != 0.1d * i) throw new Exception("unequal");

                var t4 = new List<ushort>();
                for (int i = 0; i < n; ++i) t4.Add(0);
                Campy.Parallel.For(n, i => t4[i] = (ushort)(i + 1));
                for (int i = 0; i < n; ++i) if (t4[i] != (ushort)(i + 1)) throw new Exception("unequal");

                var t5 = new List<int>();
                for (int i = 0; i < n; ++i) t5.Add(0);
                Campy.Parallel.For(n, i => t5[i] = t1[i] * 2);
                for (int i = 0; i < n; ++i) if (t5[i] != t1[i] * 2) throw new Exception("unequal");

                var t6 = new List<float>();
                for (int i = 0; i < n; ++i) t6.Add(0);
                Campy.Parallel.For(n, i => t6[i] = 0.1f * i + t2[i]);
                for (int i = 0; i < n; ++i) if (t6[i] != 0.1f * i + t2[i]) throw new Exception("unequal");

                var t7 = new List<double>();
                for (int i = 0; i < n; ++i) t7.Add(0);
                Campy.Parallel.For(n, i => t7[i] = 0.1f * i - t3[i]);
                for (int i = 0; i < n; ++i) if (t7[i] != 0.1f * i - t3[i]) throw new Exception("unequal");

                var t8 = new List<ushort>();
                for (int i = 0; i < n; ++i) t8.Add(0);
                Campy.Parallel.For(n, (Index i) =>
                {
                    t8[i] = (ushort)(t4[i] + i + 1);
                });
                for (int i = 0; i < n; ++i) if (t8[i] != (ushort)(t4[i] + i + 1)) throw new Exception("unequal");
            }
            {
                int n = 4;
                int[] x = new int[n];
                Campy.Parallel.For(n, i => x[i] = i);
            }
            //{
            //    int n = 4;
            //    int[] x = new int[n];
            //    Campy.Parallel.For(n, i => x[i[0]] = i[0]);
            //}
            {
                int n = 4;
                System.UInt16[] t4 = new ushort[n];
                Campy.Parallel.For(n, i => t4[i] = (ushort)(i + 1));
                for (int i = 0; i < n; ++i) if (t4[i] != (ushort)(i + 1)) throw new Exception("unequal");
                System.UInt16[] t8 = new ushort[n];
                Campy.Parallel.For(n, (Index i) =>
                {
                    t8[i] = (ushort)(t4[i] + i + 1);
                });
            }
            {
                int[][] jagged_array = new int[][]
                {
                    new int[] {1, 3, 5, 7, 9},
                    new int[] {0, 2, 4, 6},
                    new int[] {11, 22}
                };
                System.Console.WriteLine(jagged_array[0][0]);
                System.Console.WriteLine(jagged_array[0][1]);
                System.Console.WriteLine(jagged_array[0][2]);
                System.Console.WriteLine(jagged_array[0].Length);
                System.Console.WriteLine(jagged_array[1].Length);
                System.Console.WriteLine(jagged_array[2].Length);

                Campy.Parallel.For(3, i =>
                {
                    jagged_array[i][0] = i+43; //jagged_array[i].Length;
                });
            }

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
            //    // Saxpy (vector update).
            //    int n = 2;
            //    float[] x = new float[n];
            //    float[] y = new float[n];
            //    float a = 10f;

            //    Campy.Parallel.For(n, i => x[i] = i);
            //    Campy.Parallel.For(n, i => y[i] = i - 1);
            //    Campy.Parallel.For(n, i =>
            //    {
            //        y[i] = y[i] + a * x[i];
            //    });
            //    for (int i = 0; i < n; ++i) System.Console.Write(y[i] + " ");
            //    System.Console.WriteLine();
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
