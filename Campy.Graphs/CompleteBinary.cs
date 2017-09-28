namespace Campy.Graphs
{
    public class CompleteBinary : BinaryTreeLinkList<int>
    {
        int MakeSize;
        int counter;

        public CompleteBinary()
            : base()
        {
            this.MakeSize = 8;
            this.SpecialCreate();
            //base.Sanity();
        }

        public CompleteBinary(int size)
            : base()
        {
            this.MakeSize = size;
            this.SpecialCreate();
            //base.Sanity();
        }

        void SpecialCreate()
        {
            counter = 0;
            BinaryTreeLinkList<int>.Vertex r = (BinaryTreeLinkList<int>.Vertex)this.AddVertex(counter++);
            this._Root = r;
            MakeIt(this.MakeSize, r);
        }

        void MakeIt(int current_height, BinaryTreeLinkList<int>.Vertex current_node)
        {
            if (current_height == 0)
                return;
            current_height--;
            BinaryTreeLinkList<int>.Vertex l = (BinaryTreeLinkList<int>.Vertex)this.AddVertex(counter++);
            BinaryTreeLinkList<int>.Vertex r = (BinaryTreeLinkList<int>.Vertex)this.AddVertex(counter++);
            this.AddEdge(current_node, l);
            MakeIt(current_height, current_node.Left);
            this.AddEdge(current_node, r);
            MakeIt(current_height, current_node.Right);
        }
    }
}
