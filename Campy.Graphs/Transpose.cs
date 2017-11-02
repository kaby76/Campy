namespace Campy.Graphs
{
    public class Transpose<T,E>
        where E : IEdge<T>
    {
        public static IGraph<T,E> getTranspose(IGraph<T,E> graph)
        {
            IGraph<T,E> g = new GraphAdjList<T,E>();
            foreach (var v in graph.Vertices) g.AddVertex(v);
            foreach (var e in graph.Edges) g.AddEdge(e);
            return g;
        }
    }
}
