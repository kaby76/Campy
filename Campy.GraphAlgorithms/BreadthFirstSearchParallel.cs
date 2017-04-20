using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Campy.Graphs;

namespace Campy.GraphAlgorithms
{
    public class BreadthFirstSearchParallel<NAME>
    {
        IGraph<NAME> graph;
        IEnumerable<NAME> Source;
        Dictionary<NAME, bool> Visited = new Dictionary<NAME, bool>();

        public BreadthFirstSearchParallel(IGraph<NAME> g, IEnumerable<NAME> s)
        {
            graph = g;
            Source = s;
            foreach (NAME v in graph.Vertices)
                Visited.Add(v, false);
        }

        bool Terminate = false;

        public void VisitNodes(Func<NAME, bool> func, bool backwards = false)
        {
            if (Source.Count() != 0)
            {
                // Accumulate all vertices since yield return cannot be used in
                // a lambda function.
                Queue<NAME> all_nodes_in_order = new Queue<NAME>();
                Queue<NAME> current_frontier = new Queue<NAME>();
                Queue<NAME> next_frontier = new Queue<NAME>();
                Object thisLock = new Object();

                Parallel.ForEach(graph.Vertices, (NAME v) =>
                {
                    Visited[v] = false;
                });

                foreach (NAME v in Source)
                    current_frontier.Enqueue(v);

                while (current_frontier.Count != 0)
                {
                    Parallel.ForEach(current_frontier, (NAME u) =>
                    {
                        //yield return u; unfortunately, yield return 
                        // cannot be used in a lambda function.
                        Visited[u] = true;
                        NAME uu = u;
                        bool term = func(uu);
                        if (term)
                        {
                            Terminate = true;
                            return;
                        }

                        foreach (NAME v in graph.Successors(u))
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
                    next_frontier = new Queue<NAME>();
                }
            }
        }
    }
}
