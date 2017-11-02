using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Campy.Graphs
{
    public class Tarjan<T, E> : IEnumerable<T>
        where E : IEdge<T>
    {
        private Dictionary<T, bool> visited = new Dictionary<T, bool>();
        private Dictionary<T, bool> closed = new Dictionary<T, bool>();
        private IGraph<T, E> _graph;
        int index = 0; // number of nodes
        Stack<T> S = new Stack<T>();
        Dictionary<T, int> Index = new Dictionary<T, int>();
        Dictionary<T, int> LowLink = new Dictionary<T, int>();

        public Tarjan(IGraph<T,E> graph)
        {
            _graph = graph;
            foreach (var v in _graph.Vertices)
            {
                Index[v] = -1;
                LowLink[v] = -1;
            }
        }

        IEnumerable<T> StrongConnect(T v)
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

                T w;
                do
                {
                    w = S.Pop();
                  Console.Write(w + " ");
                    yield return w;
                } while (!w.Equals(v));

               Console.WriteLine();
            }
        }

        public IEnumerable<T> GetEnumerable()
        {
            foreach (var v in _graph.Vertices)
            {
                if (_graph.Predecessors(v).Any()) continue;
                if (Index[v] < 0)
                    foreach (var w in StrongConnect(v))
                        yield return w;
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            foreach (var v in _graph.Vertices)
            {
                if (_graph.Predecessors(v).Any()) continue;
                if (Index[v] < 0)
                    foreach (var w in StrongConnect(v))
                        yield return w;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
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
