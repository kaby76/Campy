using System;

namespace Campy.Graphs
{
    public interface IEdge<NAME> : IComparable<IEdge<NAME>>
    {
        NAME From
        {
            get;
        }

        NAME To
        {
            get;
        }
    }
}
