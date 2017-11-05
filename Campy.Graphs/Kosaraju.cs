using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Campy.Graphs
{
    public class Kosaraju
    {
        private void fillOrder<T,E>(IGraph<T, E> gr, T v, Dictionary<T, bool> visited, Stack<T> stack)
            where E : IEdge<T>
        {
            // Mark the current node as visited and print it
            visited[v] = true;

            foreach (var n in gr.Successors(v))
            {
                if (!visited[n])
                    fillOrder(gr, n, visited, stack);
            }

            stack.Push(v);
        }

        // A recursive function to print DFS starting from v
        private List<T> DFSUtil<T,E>(List<T> result, IGraph<T, E> g, T v, Dictionary<T, bool> visited)
            where E : IEdge<T>
        {
            // Mark the current node as visited and print it
            visited[v] = true;

            result.Add(v);

            // For all predecessors and successors to v...
            foreach (var n in g.Successors(v))
            {
                if (!visited[n])
                    DFSUtil(result, g, n, visited);
            }
            return result;
        }

         public IEnumerator<List<T>> GetSccs<T,E>(IGraph<T, E> graph)
            where E : IEdge<T>
        {
            Stack<T> stack = new Stack<T>();

            Dictionary<T, bool> visited = new Dictionary<T, bool>();
            foreach (var i in graph.Vertices)
                visited[i] = false;

            foreach (var i in graph.Vertices)
                if (visited[i] == false)
                    fillOrder(graph, i, visited, stack);

            var gr = Transpose<T,E>.getTranspose(graph);

            foreach (var i in gr.Vertices)
                visited[i] = false;

            while (stack.Any())
            {
                // Pop a vertex from stack
                var v = stack.Pop();

                // Print Strongly connected component of the popped vertex
                if (visited[v] == false)
                {
                    List<T> result = new List<T>();
                    yield return DFSUtil(result, gr, v, visited);
                }
            }
        }
    }
}