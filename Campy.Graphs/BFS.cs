using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Campy.Graphs
{
    public class BFS<T, E> : IEnumerable<T>
        where E : IEdge<T>
    {

        IGraph<T,E> graph;
        IEnumerable<T> Source;
        bool _backwards;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();

        public BFS(IGraph<T,E> g, IEnumerable<T> s, bool backwards = false)
        {
            graph = g;
            Source = s;
            _backwards = backwards;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        public BFS(IGraph<T,E> g, T s)
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

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
