using System.Collections.Generic;

namespace Campy.Graphs
{
    public interface IGraph<NODE, EDGE>
        where EDGE : IEdge<NODE>
    {
        IEnumerable<NODE> Vertices
        {
            get;
        }

        IEnumerable<EDGE> Edges
        {
            get;
        }

        NODE AddVertex(NODE v);

        EDGE AddEdge(EDGE e);

        IEnumerable<NODE> Predecessors(NODE n);

        IEnumerable<EDGE> PredecessorEdges(NODE n);

        IEnumerable<NODE> ReversePredecessors(NODE n);

        IEnumerable<NODE> Successors(NODE n);

        IEnumerable<EDGE> SuccessorEdges(NODE n);

        IEnumerable<NODE> ReverseSuccessors(NODE n);

        bool IsLeaf(NODE node);
    }
}
