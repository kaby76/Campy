namespace Campy.Graphs
{
    public class CompleteBinary : BinaryTreeAdjList<int, DirectedEdge<int>>
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
            var r = this.AddVertex(counter++);
            this._Root = r;
            MakeIt(this.MakeSize, r);
        }

        void MakeIt(int current_height, int current_node)
        {
            if (current_height == 0)
                return;
            current_height--;
            int l = this.AddVertex(counter++);
            int r = this.AddVertex(counter++);
            this.AddEdge(new DirectedEdge<int>(current_node, l));
            MakeIt(current_height, this.Left(current_node));
            this.AddEdge(new DirectedEdge<int>(current_node, r));
            MakeIt(current_height, this.Right(current_node));
        }
    }
}
