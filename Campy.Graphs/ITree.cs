namespace Campy.Graphs
{
    public interface ITree<T> : IGraph<T, DirectedEdge<T>>
        where T: ITreeVertex<T>
    {
        T _Root
        {
            get;
            set;
        }
    }
}
