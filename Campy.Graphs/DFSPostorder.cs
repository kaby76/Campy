using System.Collections;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    // Algorithms adapted from "A NEW NON-RECURSIVE ALGORITHM FOR
    // BINARY SEARCH TREE TRAVERSAL", Akram Al-Rawi, Azzedine Lansari, Faouzi Bouslama
    // N.B.: There is no "in-order" traversal defined for a general graph,
    // it must be a binary tree.
    public class DFSPostorder<T, E> : IEnumerable<T>
        where E : IEdge<T>
    {
        IGraph<T,E> graph;
        IEnumerable<T> Source;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();

        public DFSPostorder(IGraph<T,E> g, IEnumerable<T> s)
        {
            graph = g;
            Source = s;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        StackQueue<T> Stack = new StackQueue<T>();

        public System.Collections.Generic.IEnumerator<T> GetEnumerator()
        {
            foreach (T v in graph.Vertices)
                Visited[v] = false;

            foreach (T v in Source)
                Stack.Push(v);

            while (Stack.Count != 0)
            {
                T u = Stack.Pop();
                if (Visited[u])
                {
                    yield return u;
                }
                else
                {
                    Visited[u] = true;
                    Stack.Push(u);
                    foreach (T v in graph.ReverseSuccessors(u))
                    {
                        if (!Visited[v] && !Stack.Contains(v))
                            Stack.Push(v);
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
