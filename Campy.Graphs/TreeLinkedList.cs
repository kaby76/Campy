using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace Campy.Graphs
{

    public class TreeLinkedList<NAME, NODE, EDGE> : GraphLinkedList<NAME, NODE, EDGE>
        where NODE : TreeLinkedList<NAME, NODE, EDGE>.Vertex, new()
        where EDGE : TreeLinkedList<NAME, NODE, EDGE>.Edge, new()
    {
        /// <summary>
        /// Constructor, and creates a predefined tree (see below).
        /// </summary>
        public TreeLinkedList()
            : base()
        {
        }

        //public void Sanity()
        //{
        //    // Check sanity of a complete tree.
        //    // 1. Check there is one and only one root.
        //    int count_roots = 0;
        //    foreach (Vertex v in this.Vertices)
        //    {
        //        Vertex tv = v as Vertex;
        //        if (tv.Parent == null)
        //            count_roots++;
        //    }
        //    if (count_roots != 1 && this.Vertices.Count() != 0)
        //        throw new Exception("Tree malformed -- there are " + count_roots + " roots.");

        //    // 2. Check each node that is has one parent, except for root, and that it's equal to predecessor list.
        //    foreach (Vertex v in this.Vertices)
        //    {
        //        if (v.Parent == null && this.Root != v)
        //            throw new Exception("Tree malformed -- node without a parent that isn't the root.");
        //        if (v.Predecessors.Count() > 1)
        //            throw new Exception("Tree malformed -- predecessor count greater than one.");
        //        if (v.Predecessors.Count() == 0 && this.Root != v)
        //            throw new Exception("Tree malformed -- node without a parent that isn't the root.");
        //        if (v.Predecessors.Count() != 0 && v.Predecessors.First() != v.Parent)
        //            throw new Exception("Tree malformed -- node predecessor and parent are inconsistent.");
        //    }

        //    // 3. Go through edge list and verify.
        //    Dictionary<GraphLinkedList<NAME, NODE, EDGE>.Vertex, bool> seen = new Dictionary<GraphLinkedList<NAME, NODE, EDGE>.Vertex, bool>();
        //    foreach (Edge e in this.Edges)
        //    {
        //        if (!seen.ContainsKey(e.To))
        //            seen.Add(e.To, true);
        //        else
        //        {
        //            throw new Exception("Tree malformed -- Visited more than once.");
        //        }
        //    }
        //}

        protected Vertex _Root;

        public NAME Root
        {
            get { return _Root.Name; }
            set
            {
                int id = this.NameSpace.BijectFromBasetype(value);
                NODE node = this.VertexSpace[id];
                _Root = node;
            }
        }

        //public GraphLinkedList<NAME> LeftContour(Vertex<NAME> v)
        //{
        //    // Create a graph with the left contour of v.
        //    GraphLinkedList<NAME> lc = new GraphLinkedList<NAME>();

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

        //public GraphLinkedList<NAME> RightContour(Vertex<NAME> v)
        //{
        //    // Create a graph with the right contour of v.
        //    GraphLinkedList<NAME> rc = new GraphLinkedList<NAME>();

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

        int height(GraphLinkedList<NAME, NODE, EDGE>.Vertex v, int d)
        {
            if (v == null)
                return d;
            int m = d;
            foreach (Edge e in v._Successors)
            {
                GraphLinkedList<NAME, NODE, EDGE>.Vertex u = e.to;
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

        new public class Vertex : GraphLinkedList<NAME, NODE, EDGE>.Vertex
        {
            public Vertex()
                : base()
            {
            }

            public Vertex(NAME t)
                : base(t)
            {
            }
        }

        new public class Edge : GraphLinkedList<NAME, NODE, EDGE>.Edge
        {
            public Edge()
            {
            }

            public Edge(GraphLinkedList<NAME, NODE, EDGE>.Vertex f, GraphLinkedList<NAME, NODE, EDGE>.Vertex t)
            {
                from = f;
                to = t;
            }
        }
    }

    public class TreeLinkedListVertex<NAME>
        : TreeLinkedList<NAME, TreeLinkedListVertex<NAME>, TreeLinkedListEdge<NAME>>.Vertex
    {
    }

    public class TreeLinkedListEdge<NAME>
        : TreeLinkedList<NAME, TreeLinkedListVertex<NAME>, TreeLinkedListEdge<NAME>>.Edge
    {
    }

    public class TreeLinkedList<NAME>
        : TreeLinkedList<NAME,
            TreeLinkedListVertex<NAME>,
            TreeLinkedListEdge<NAME>>
    {
    }
}
