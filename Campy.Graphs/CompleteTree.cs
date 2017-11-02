using System;
using System.Collections.Generic;
using System.Linq;

namespace Campy.Graphs
{
    public class CompleteTree : TreeAdjList<int, DirectedEdge<int>>
    {
        int Depth;
        int Expansion;
        int counter;

        public CompleteTree()
            : base()
        {
            this.Expansion = 4;
            this.Depth = 8;
            this.SpecialCreate();
            //base.Sanity();
        }

        public CompleteTree(int expansion, int depth)
            : base()
        {
            this.Expansion = expansion;
            this.Depth = depth;
            this.SpecialCreate();
            //base.Sanity();
        }

        void SpecialCreate()
        {
            // Set up range.
            int top = 1;
            for (int i = 0; i < Depth + 1; ++i)
                top = top * Expansion;

            counter = 0;
            int r = this.AddVertex(counter++);
            this.Root = 0;
            Stack<Tuple<int, int>> stack = new Stack<Tuple<int, int>>();
            stack.Push(new Tuple<int, int>(r, 0));
            while (stack.Count != 0)
            {
                Tuple<int, int> t = stack.Pop();
                int v = t.Item1;
                int level = t.Item2;
                level += 1;
                for (int i = Expansion; i > 0; --i)
                {
                    int c = this.AddVertex(counter++);
                    this.AddEdge(new DirectedEdge<int>(v, c));
                    if (level < Depth)
                        stack.Push(new Tuple<int, int>(c, level));
                }
            }
        }
    }
}
