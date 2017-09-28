using System.Collections.Generic;

namespace Campy.Graphs
{
    public interface ITreeVertex<T> : IVertex<T>
    {
        List<IVertex<T>> Children
        {
            get;
            set;
        }

        IVertex<T> Parent
        {
            get;
        }
    }
}
