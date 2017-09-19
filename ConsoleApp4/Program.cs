using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Linq;
using Campy.Types;
using Campy.Utils;

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
            //Campy.Utils.Options.Set("graph_trace", true);
            //Campy.Utils.Options.Set("module_trace", true);
            //Campy.Utils.Options.Set("name_trace", true);
            //Campy.Utils.Options.Set("cfg_construction_trace", true);
            //Campy.Utils.Options.Set("dot_graph", true);
            //Campy.Utils.Options.Set("jit_trace", true);

            List<int> x = new List<int>();
            for (int i = 0; i < 4; ++i) x.Add(0);
            Campy.Parallel.For(new Extent(4), i =>
            {
                x[i] = i;
            });
            foreach (var e in x)
                System.Console.Write(e + " ");
            System.Console.WriteLine();

            int max_level = 16;
            int n = Bithacks.Power2(max_level);
           // int[] data = new int[n];
            List<int> data = Enumerable.Repeat(0, n).ToList();

            Campy.Parallel.For(new Extent(n), idx => data[idx] = 1);
            for (int level = 1; level <= Bithacks.Log2(n); level++)
            {
                int step = Bithacks.Power2(level);
                Campy.Parallel.For(new Extent(n / step), idx =>
                {
                    var i = step * idx;
                    data[i] = data[i] + data[i + step / 2];
                });
                System.Console.WriteLine("level " + level);
                //for (int i = 0; i < data.Count; ++i)
                //    System.Console.Write(data[i] + " ");
                //System.Console.WriteLine();
            }

            //for (int i = 0; i < data.Count; ++i) System.Console.Write(data[i] + " ");
            //System.Console.WriteLine();
            System.Console.WriteLine("sum = " + data[0]);

            // Create complete binary tree.
            //int max_level = 6;
            //N root = new N();
            //counter++;
            //List<N> all_nodes = new List<N>();
            //all_nodes.Add(root);
            //MakeIt(max_level, root, ref all_nodes);
            //root.visit = true;
            //int size = all_nodes.Count;
            //for (;;)
            //{
            //    bool changed = false;
            //    for (int i = 0; i < size; ++i)
            //    {
            //        if (i >= size)
            //            continue;
            //        N node = all_nodes[i];
            //        if (!node.visit)
            //            continue;
            //        node.visit = false;
            //        node.visited = true;
            //        N l = node.Left;
            //        N r = node.Right;
            //        if (l != null)
            //        {
            //            l.visit = true;
            //            l.level = node.level + 1;
            //            changed = true;
            //        }
            //        if (r != null)
            //        {
            //            r.visit = true;
            //            r.level = node.level + 1;
            //            changed = true;
            //        }
            //    }
            //    if (!changed)
            //        break;
            //}
            //Campy.Parallel.For(new Extent(size), i
            //    =>
            //{
            //    if (i >= size) return;
            //    N node = all_nodes[i];
            //    if (!node.visit)
            //        return;
            //    node.visit = false;
            //    node.visited = true;
            //    N l = node.Left;
            //    N r = node.Right;
            //    if (l != null)
            //    {
            //        l.visit = true;
            //        l.level = node.level + 1;
            //    }
            //    if (r != null)
            //    {
            //        r.visit = true;
            //        r.level = node.level + 1;
            //    }
            //});

            //for (int level = 0; level < max_level; ++level)
            //{
            //    Campy.Parallel.For(new Extent(Bithacks.Power2(level)), i =>
            //    {

            //    });
            //}
        }
    }
}
