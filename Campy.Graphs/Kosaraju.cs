using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Campy.Graphs
{
    public class Kosaraju<T, E> : IEnumerable<T>
        where E : IEdge<T>
    {
        private IGraph<T, E> _graph;

        public Kosaraju(IGraph<T, E> graph)
        {
            _graph = graph;
        }

        private void fillOrder(IGraph<T, E> gr, T v, Dictionary<T, bool> visited, Stack<T> stack)
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



        public class SCCEnumerator
        {
            private IGraph<T, E> graph;
            private T start;
            private Dictionary<T, bool> visited;

            public SCCEnumerator(IGraph<T, E> g, T s, Dictionary<T, bool> v)
            {
                graph = g;
                start = s;
                visited = v;
            }

            // A recursive function to print DFS starting from v
            private List<T> DFSUtil(List<T> result, IGraph<T, E> g, T v, Dictionary<T, bool> visited)
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

            public List<T> ToList()
            {
                return DFSUtil(new List<T>(), graph, start, visited);
            }
        }


        public IEnumerator<List<T>> GetEnumerator()
        {
            Stack<T> stack = new Stack<T>();

            Dictionary<T, bool> visited = new Dictionary<T, bool>();
            foreach (var i in _graph.Vertices)
                visited[i] = false;

            foreach (var i in _graph.Vertices)
                if (visited[i] == false)
                    fillOrder(_graph, i, visited, stack);

            var gr = Transpose<T,E>.getTranspose(_graph);

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
                    yield return new SCCEnumerator(gr, v, visited).ToList();
                }
            }
        }

        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            throw new NotImplementedException();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}