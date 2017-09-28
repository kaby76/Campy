namespace Campy.Graphs
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
