using System;

namespace Campy.Graphs
{
    public interface IVertex<NAME> : IComparable<IVertex<NAME>>
    {
        NAME Name
        {
            get;
        }
    }
}
