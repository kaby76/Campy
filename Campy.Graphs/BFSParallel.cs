using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Campy.Graphs
{
    public class BFSParallel<T,E>
        where E : IEdge<T>
    {
        IGraph<T,E> graph;
        IEnumerable<T> Source;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();

        public BFSParallel(IGraph<T,E> g, IEnumerable<T> s)
        {
            graph = g;
            Source = s;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        public void VisitNodes(Func<T, bool> func, bool backwards = false)
        {
            if (Source.Count() != 0)
            {
                // Accumulate all vertices since yield return cannot be used in
                // a lambda function.
                Queue<T> all_nodes_in_order = new Queue<T>();
                Queue<T> current_frontier = new Queue<T>();
                Queue<T> next_frontier = new Queue<T>();
                Object thisLock = new Object();

                Parallel.ForEach(graph.Vertices, (T v) =>
                {
                    Visited[v] = false;
                });

                foreach (T v in Source)
                    current_frontier.Enqueue(v);

                while (current_frontier.Count != 0)
                {
                    Parallel.ForEach(current_frontier, (T u) =>
                    {
                        //yield return u; unfortunately, yield return 
                        // cannot be used in a lambda function.
                        Visited[u] = true;
                        T uu = u;
                        bool term = func(uu);
                        if (term)
                        {
                            return;
                        }

                        foreach (T v in graph.Successors(u))
                        {
                            if (!Visited[v])
                            {
                                Visited[v] = true;
                                lock (thisLock)
                                {
                                    next_frontier.Enqueue(v);
                                }
                            }
                        };
                    });
                    current_frontier = next_frontier;
                    next_frontier = new Queue<T>();
                }
            }
        }
    }
}
