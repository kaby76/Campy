using System.Collections;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    // Algorithms adapted from "A NEW NON-RECURSIVE ALGORITHM FOR
    // BINARY SEARCH TREE TRAVERSAL", Akram Al-Rawi, Azzedine Lansari, Faouzi Bouslama
    // N.B.: There is no "in-order" traversal defined for a general graph,
    // it must be a binary tree.
    public class DFSInorder<T, E> : IEnumerable<T>
        where E : IEdge<T>
    {
        BinaryTreeAdjList<T,E> graph;
        T Source = default(T);
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();

        public DFSInorder(BinaryTreeAdjList<T,E> g, T s)
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

            for (T s = Source; s != null; )
            {
                Stack.Push(s);
                T ver = graph.Left(s);
                s = ver;
            }

            while (Stack.Count != 0)
            {
                T u = Stack.Pop();
                yield return u;

                T ver = u;
                u = graph.Right(ver);
                while (u != null)
                {
                    Stack.Push(u);
                    T z = u;
                    u = graph.Right(ver);
                    u = graph.Left(z);
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
