using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Campy.Graphs;

namespace Campy.GraphAlgorithms
{
    public class Transpose<NAME>
    {
        public static IGraph<NAME> getTranspose(IGraph<NAME> graph)
        {
            IGraph<NAME> g = new GraphLinkedList<NAME>();
            foreach (var v in graph.Vertices) g.AddVertex(v);
            foreach (var e in graph.Edges) g.AddEdge(e.To, e.From);
            return g;
        }
    }
}
