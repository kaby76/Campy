using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections;

namespace Campy.Graphs
{

    /// <summary>
    /// This class enumerates a topological sort of a graph. If the graph
    /// is not a DAG, then it will fail. Note, check for DAGs before running this
    /// algorithm using Kosaraju or Tarjan.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class TopologicalSort
    {

        /// Topological Sorting (Kahn's algorithm) 
        public IEnumerator<T> Sort<T,E>(IGraph<T, E> graph, T s)
            where E : IEdge<T>
        {
            var source = new T[] { s };
            Dictionary<T, bool> visited = new Dictionary<T, bool>();

            foreach (T v in graph.Vertices)
                visited.Add(v, false);

            if (source != null && source.Any())
            {
                HashSet<T> nodes = new HashSet<T>();
                foreach (T v in graph.Vertices) nodes.Add(v);
                HashSet<Tuple<T, T>> edges = new HashSet<Tuple<T, T>>();
                foreach (IEdge<T> e in graph.Edges) edges.Add(new Tuple<T, T>(e.From, e.To));
                
                // Set of all nodes with no incoming edges
                var S = new HashSet<T>(nodes.Where(n => edges.All(e => e.Item2.Equals(n) == false)));

                // while S is non-empty do
                while (S.Any())
                {

                    //  remove a node n from S
                    var n = S.First();
                    S.Remove(n);

                    // add n to tail of L
                    yield return n;

                    // for each node m with an edge e from n to m do
                    var look = edges.Where(e => e.Item1.Equals(n)).ToList();
                    foreach (var e in look)
                    {
                        var m = e.Item2;

                        // remove edge e from the graph
                        edges.Remove(e);

                        // if m has no other incoming edges then
                        if (edges.All(me => me.Item2.Equals(m) == false))
                        {
                            // insert m into S
                            S.Add(m);
                        }
                    }
                }
            }
        }
    }
}

