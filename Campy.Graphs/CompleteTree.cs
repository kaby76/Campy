using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.Graphs
{
    public class CompleteTree : TreeAdjList<int>
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

            this.SetNameSpace(Enumerable.Range(0, top));
            counter = 0;
            TreeAdjListVertex<int> r = (TreeAdjListVertex<int>)this.AddVertex(counter++);
            this.Root = 0;
            Stack<Tuple<TreeAdjListVertex<int>, int>> stack = new Stack<Tuple<TreeAdjListVertex<int>, int>>();
            stack.Push(new Tuple<TreeAdjListVertex<int>, int>(r, 0));
            while (stack.Count != 0)
            {
                Tuple<TreeAdjListVertex<int>, int> t = stack.Pop();
                TreeAdjListVertex<int> v = t.Item1;
                int level = t.Item2;
                level += 1;
                for (int i = Expansion; i > 0; --i)
                {
                    TreeAdjListVertex<int> c = (TreeAdjListVertex<int>)this.AddVertex(counter++);
                    this.AddEdge(v.Name, c.Name);
                    if (level < Depth)
                        stack.Push(new Tuple<TreeAdjListVertex<int>, int>(c, level));
                }
            }
        }
    }
}
