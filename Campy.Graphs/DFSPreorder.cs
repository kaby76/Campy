using System.Collections;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    // Algorithms adapted from "A NEW NON-RECURSIVE ALGORITHM FOR
    // BINARY SEARCH TREE TRAVERSAL", Akram Al-Rawi, Azzedine Lansari, Faouzi Bouslama
    // N.B.: There is no "in-order" traversal defined for a general graph,
    // it must be a binary tree.
    public class DFSPreorder<T> : IEnumerable<T>
    {
        IGraph<T> graph;
        IEnumerable<T> Source;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();
        StackQueue<T> Stack = new StackQueue<T>();

        public DFSPreorder(IGraph<T> g, IEnumerable<T> s)
        {
            graph = g;
            Source = s;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        public System.Collections.Generic.IEnumerator<T> GetEnumerator()
        {
            foreach (T v in graph.Vertices)
                Visited[v] = false;

            foreach (T v in Source)
                Stack.Push(v);

            while (Stack.Count != 0)
            {
                T u = Stack.Pop();
                Visited[u] = true;
                yield return u;
                foreach (T v in graph.ReverseSuccessors(u))
                {
                    if (!Visited[v] && !Stack.Contains(v))
                        Stack.Push(v);
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    public class DepthFirstPreorderTraversalViaPredecessors<T>
    {
        IGraph<T> graph;
        IEnumerable<T> Source;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();
        StackQueue<T> Stack = new StackQueue<T>();

        public DepthFirstPreorderTraversalViaPredecessors(IGraph<T> g, IEnumerable<T> s)
        {
            graph = g;
            Source = s;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        public System.Collections.Generic.IEnumerator<T> GetEnumerator()
        {
            foreach (T v in graph.Vertices)
                Visited[v] = false;

            foreach (T v in Source)
                Stack.Push(v);

            while (Stack.Count != 0)
            {
                T u = Stack.Pop();
                Visited[u] = true;
                yield return u;
                foreach (T v in graph.ReversePredecessors(u))
                {
                    if (!Visited[v] && !Stack.Contains(v))
                        Stack.Push(v);
                }
            }
        }
    }
}
