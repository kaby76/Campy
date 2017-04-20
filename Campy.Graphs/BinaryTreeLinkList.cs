using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace Campy.Graphs
{
    public class BinaryTreeLinkList<NAME, NODE, EDGE> : TreeLinkedList<NAME, NODE, EDGE>
        where NODE : BinaryTreeLinkList<NAME, NODE, EDGE>.Vertex, new()
        where EDGE : BinaryTreeLinkList<NAME, NODE, EDGE>.Edge, new()
    {
        public BinaryTreeLinkList()
            : base()
        {
        }

        new public BinaryTreeLinkList<NAME, NODE, EDGE>.Edge AddEdge(NAME f, NAME t)
        {
            Edge e = (Edge)base.AddEdge(f, t);
            // Deal with Left/Right.
            return e;
        }

        new public class Vertex : TreeLinkedList<NAME, NODE, EDGE>.Vertex
        {
            public Vertex Left
            {
                get;
                set;
            }

            public Vertex Right
            {
                get;
                set;
            }

            public Vertex()
            {
            }

            public Vertex(NAME t)
                : base(t)
            {
            }

        }

        new public class Edge : TreeLinkedList<NAME, NODE, EDGE>.Edge
        {
            public Edge()
            {
            }

            public Edge(GraphLinkedList<NAME, NODE, EDGE>.Vertex f, GraphLinkedList<NAME, NODE, EDGE>.Vertex t)
            {
                from = f;
                to = t;
                Vertex vf = (Vertex)f;
                Vertex vt = (Vertex)t;
                // assume first in is left child, then right child.
                if (vf.Left == null)
                    vf.Left = vt;
                else
                    vf.Right = vt;
            }
        }
    }

    public class BinaryTreeLinkListVertex<NAME>
        : BinaryTreeLinkList<NAME, BinaryTreeLinkListVertex<NAME>, BinaryTreeLinkListEdge<NAME>>.Vertex
    {
    }

    public class BinaryTreeLinkListEdge<NAME>
        : BinaryTreeLinkList<NAME, BinaryTreeLinkListVertex<NAME>, BinaryTreeLinkListEdge<NAME>>.Edge
    {
    }

    public class BinaryTreeLinkList<NAME>
        : BinaryTreeLinkList<NAME,
            BinaryTreeLinkListVertex<NAME>,
            BinaryTreeLinkListEdge<NAME>>
    {
    }
}
