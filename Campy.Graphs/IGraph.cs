using System.Collections.Generic;

namespace Campy.Graphs
{
    public interface IGraph<NAME>
    {
        IEnumerable<NAME> Vertices
        {
            get;
            //set;
        }

        IEnumerable<IEdge<NAME>> Edges
        {
            get;
            //set;
        }

        IVertex<NAME> AddVertex(NAME v);

        IEdge<NAME> AddEdge(NAME f, NAME t);

        IEnumerable<NAME> Predecessors(NAME n);

        IEnumerable<IEdge<NAME>> PredecessorEdges(NAME n);

        IEnumerable<NAME> ReversePredecessors(NAME n);

        IEnumerable<NAME> Successors(NAME n);

        IEnumerable<IEdge<NAME>> SuccessorEdges(NAME n);

        IEnumerable<NAME> ReverseSuccessors(NAME n);

        bool IsLeaf(NAME node);
    }
}
