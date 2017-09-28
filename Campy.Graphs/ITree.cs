namespace Campy.Graphs
{
    public interface ITree<T> : IGraph<T>
    {
        ITreeVertex<T> _Root
        {
            get;
            set;
        }
    }
}
