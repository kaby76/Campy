using System.Collections;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    // Algorithms adapted from "A NEW NON-RECURSIVE ALGORITHM FOR
    // BINARY SEARCH TREE TRAVERSAL", Akram Al-Rawi, Azzedine Lansari, Faouzi Bouslama
    // N.B.: There is no "in-order" traversal defined for a general graph,
    // it must be a binary tree.
    public class DFSInorder<T> : IEnumerable<T>
    {
        BinaryTreeLinkList<T> graph;
        T Source = default(T);
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();

        public DFSInorder(BinaryTreeLinkList<T> g, T s)
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
                int j = graph.NameSpace.BijectFromBasetype(s);
                BinaryTreeLinkListVertex<T> ver = graph.VertexSpace[j];
                s = ver.Left.Name;
            }

            while (Stack.Count != 0)
            {
                T u = Stack.Pop();
                yield return u;
                int j = graph.NameSpace.BijectFromBasetype(u);
                BinaryTreeLinkListVertex<T> ver = graph.VertexSpace[j];
                u = ver.Right.Name;
                while (u != null)
                {
                    Stack.Push(u);
                    int k = graph.NameSpace.BijectFromBasetype(u);
                    BinaryTreeLinkListVertex<T> z = graph.VertexSpace[k];
                    u = ver.Right.Name;
                    u = z.Left.Name;
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
