using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Campy.Utils;

namespace Campy.Graphs
{
    public class BFS
    {
        public static System.Collections.Generic.IEnumerable<T> Sort<T, E>
            (IGraph<T, E> graph, IEnumerable<T> source, bool backwards = false)
            where E : IEdge<T>
        {
            Dictionary<T, bool> Visited = new Dictionary<T, bool>();

            foreach (T v in graph.Vertices)
                Visited.Add(v, false);


            if (source != null && source.Count() != 0)
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
                    IEnumerable<T> ordered_enumerator = backwards ? graph.Predecessors(u) : graph.Successors(u);
                    foreach (T v in ordered_enumerator)
                    {
                        frontier.Enqueue(v);
                    }
                }
            }
        }
    }
}
