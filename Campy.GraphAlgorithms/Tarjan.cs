using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Campy.Graphs;

namespace Campy.GraphAlgorithms
{
    public class Tarjan<NAME>
    {
        private Dictionary<NAME, bool> visited = new Dictionary<NAME, bool>();
        private Dictionary<NAME, bool> closed = new Dictionary<NAME, bool>();
        private IGraph<NAME> _graph;
        int index = 0; // number of nodes
        Stack<NAME> S = new Stack<NAME>();
        Dictionary<NAME, int> Index = new Dictionary<NAME, int>();
        Dictionary<NAME, int> LowLink = new Dictionary<NAME, int>();

        public Tarjan(IGraph<NAME> graph)
        {
            _graph = graph;
            foreach (var v in _graph.Vertices)
            {
                Index[v] = -1;
                LowLink[v] = -1;
            }
        }

        IEnumerable<NAME> StrongConnect(NAME v)
        {
            // Set the depth index for v to the smallest unused index
            Index[v] = index;
            LowLink[v] = index;

            index++;
            S.Push(v);

            // Consider successors of v
            foreach (var w in _graph.Successors(v))
                if (Index[w] < 0)
                {
                    // Successor w has not yet been visited; recurse on it
                    foreach (var x in StrongConnect(w)) yield return x;
                    LowLink[v] = Math.Min(LowLink[v], LowLink[w]);
                }
                else if (S.Contains(w))
                    // Successor w is in stack S and hence in the current SCC
                    LowLink[v] = Math.Min(LowLink[v], Index[w]);

            // If v is a root node, pop the stack and generate an SCC
            if (LowLink[v] == Index[v])
            {
                Console.Write("SCC: ");

                NAME w;
                do
                {
                    w = S.Pop();
                    Console.Write(w + " ");
                    yield return w;
                } while (!w.Equals(v));

                Console.WriteLine();
            }
        }

        public IEnumerable<NAME> GetEnumerable()
        {
            foreach (var v in _graph.Vertices)
            {
                if (_graph.Predecessors(v).Any()) continue;
                if (Index[v] < 0)
                    foreach (var w in StrongConnect(v))
                        yield return w;
            }
        }
    }
}
