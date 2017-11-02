using System.Collections.Generic;

namespace Campy.Graphs
{
    public interface ITreeVertex<T> : IVertex
    {
        List<T> Children
        {
            get;
            set;
        }

        T Parent
        {
            get;
        }
    }
}
