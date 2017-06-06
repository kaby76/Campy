using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Campy.Graphs;

namespace Campy.GraphAlgorithms
{
    public class BreadthFirstTraversal<T>
    {

        IGraph<T> graph;
        IEnumerable<T> Source;
        bool _backwards;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();

        public BreadthFirstTraversal(IGraph<T> g, IEnumerable<T> s, bool backwards = false)
        {
            graph = g;
            Source = s;
            _backwards = backwards;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        public BreadthFirstTraversal(IGraph<T> g, T s)
        {
            graph = g;
            Source = new T[] { s };
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        public IEnumerator<T> GetEnumerator()
        {
            if (Source != null && Source.Count() != 0)
            {
                foreach (T v in graph.Vertices)
                    Visited[v] = false;

                Queue <T> frontier = new Queue<T>();

                // Find all entries.
                var entries = graph.Vertices.Where(node => !graph.Predecessors(node).Any()).ToList();
                foreach (T v in entries) frontier.Enqueue(v);

                while (frontier.Count != 0)
                {
                    T u = frontier.Peek();
                    frontier.Dequeue();
                    if (Visited[u]) continue;
                    Visited[u] = true;
                    yield return u;
                    IEnumerable<T> ordered_enumerator = _backwards ? graph.Predecessors(u) : graph.Successors(u);
                    foreach (T v in ordered_enumerator)
                    {
                        frontier.Enqueue(v);
                    }
                }
            }
        }
    }
}
