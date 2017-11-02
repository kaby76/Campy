namespace Campy.Graphs
{

    public class TreeAdjList<NODE, EDGE> : GraphAdjList<NODE, EDGE>
        where EDGE : IEdge<NODE>
    {
        /// <summary>
        /// Constructor, and creates a predefined tree (see below).
        /// </summary>
        public TreeAdjList()
            : base()
        {
        }

        protected NODE _Root;

        public NODE Root
        {
            get { return _Root; }
            set { _Root = value; }
        }

        //public GraphAdjList<NAME> LeftContour(Vertex<NAME> v)
        //{
        //    // Create a graph with the left contour of v.
        //    GraphAdjList<NAME> lc = new GraphAdjList<NAME>();

        //    // Create left contour.
        //    int llevel = 1;
        //    Vertex<NAME> left = v.GetLeftMost(0, llevel);
        //    Vertex<NAME> cloneleft = lc.CloneVertex(v);
        //    Vertex<NAME> llast = v;
        //    Vertex<NAME> clonellast = cloneleft;
        //    while (left != null)
        //    {
        //        cloneleft = lc.CloneVertex((Vertex<NAME>)left);
        //        lc.AddEdge(clonellast, cloneleft);

        //        llevel++;
        //        llast = left;
        //        clonellast = cloneleft;
        //        left = v.GetLeftMost(0, llevel);
        //    }

        //    return lc;
        //}

        //public GraphAdjList<NAME> RightContour(Vertex<NAME> v)
        //{
        //    // Create a graph with the right contour of v.
        //    GraphAdjList<NAME> rc = new GraphAdjList<NAME>();

        //    rc.CloneVertex(v);

        //    // Create right contour.
        //    int rlevel = 1;
        //    Vertex<NAME> right = v.GetRightMost(0, rlevel);
        //    Vertex<NAME> cloneright = rc.CloneVertex(v);
        //    Vertex<NAME> rlast = v;
        //    Vertex<NAME> clonerlast = cloneright;
        //    while (right != null)
        //    {
        //        cloneright = rc.CloneVertex((Vertex<NAME>)right);
        //        rc.AddEdge(clonerlast, cloneright);

        //        rlevel++;
        //        rlast = right;
        //        clonerlast = cloneright;
        //        right = v.GetRightMost(0, rlevel);
        //    }
        //    return rc;
        //}

        int height(NODE v, int d)
        {
            if (v == null)
                return d;
            int m = d;
            foreach (EDGE e in this.SuccessorEdges(v))
            {
                NODE u = e.To;
                int x = this.height(u, d + 1);
                if (x > m)
                    m = x;
            }
            return m;
        }

        public int Height()
        {
            return this.height(this._Root, 1);
        }

        //public Vertex<NAME> CloneVertex(Vertex<NAME> other)
        //{
        //    Vertex<NAME> v = new Vertex<NAME>();
        //    v.Copy(other);
        //    if (v.Parent == null)
        //        this.Root = v;
        //    return v;
        //}
    }
}
