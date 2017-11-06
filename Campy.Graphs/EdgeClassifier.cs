using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Campy.Utils;

namespace Campy.Graphs
{
    public class EdgeClassifier
    {
        enum Color
        {
            Black,
            Gray,
            White
        }

        public enum Classification
        {
            Tree,
            Forward,
            Back,
            Cross
        }

        private static void Visit<T, E>(
            T u,
            IGraph<T, E> graph,
            Dictionary<T, Color> color,
            Dictionary<T, int> d,
            Dictionary<T, int> f,
            Dictionary<E, Classification> classify,
            ref int time)
            where E : IEdge<T>
        {
            color[u] = Color.Gray;
            time = time + 1;
            d[u] = time;
            foreach (var e in graph.SuccessorEdges(u))
            {
                if (color[e.To] == Color.White)
                {
                    classify[e] = Classification.Tree;
                    Visit(e.To, graph, color, d, f, classify, ref time);
                }
                else if (color[e.To] == Color.Gray)
                    classify[e] = Classification.Back;
                else if (color[e.To] == Color.Black)
                {
                    if (d[u] < d[e.To])
                        classify[e] = Classification.Forward;
                    else classify[e] = Classification.Cross;
                }
            }

            color[u] = Color.Black;
            time = time + 1;
            f[u] = time;
        }

        public static void Classify<T, E>
            (IGraph<T, E> graph, T u, ref Dictionary<E, Classification> classify)
            where E : IEdge<T>
        {
            Dictionary<T, Color> color = new Dictionary<T, Color>();
            Dictionary<T, int> d = new Dictionary<T, int>();
            Dictionary<T, int> f = new Dictionary<T, int>();

            foreach (var v in graph.Vertices)
                color[v] = Color.White;

            int time = 0;
            Visit(u, graph, color, d, f, classify, ref time);
            foreach (var v in graph.Vertices)
                if (color[v] == Color.White)
                    Visit(v, graph, color, d, f, classify, ref time);
        }
    }
}
