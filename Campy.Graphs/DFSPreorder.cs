using System.Collections;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    // Algorithms adapted from "A NEW NON-RECURSIVE ALGORITHM FOR
    // BINARY SEARCH TREE TRAVERSAL", Akram Al-Rawi, Azzedine Lansari, Faouzi Bouslama
    // N.B.: There is no "in-order" traversal defined for a general graph,
    // it must be a binary tree.
    public class DFSPreorder
    {
        public static System.Collections.Generic.IEnumerable<T> Sort<T, E>
            (IGraph<T, E> graph, IEnumerable<T> source)
            where E : IEdge<T>
        {
            Dictionary<T, bool> Visited = new Dictionary<T, bool>();
            StackQueue<T> Stack = new StackQueue<T>();

            foreach (T v in graph.Vertices)
                Visited[v] = false;

            foreach (T v in source)
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
    }

    public class DFSPreorderPredecessors
    {
        public static System.Collections.Generic.IEnumerable<T> Sort<T, E>
            (IGraph<T, E> graph, IEnumerable<T> source)
            where E : IEdge<T>
        {
            Dictionary<T, bool> Visited = new Dictionary<T, bool>();
            StackQueue<T> Stack = new StackQueue<T>();

            foreach (T v in graph.Vertices)
                Visited[v] = false;

            foreach (T v in source)
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
