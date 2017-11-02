using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace Campy.Graphs
{
    public class BinaryTreeAdjList<NODE, EDGE> : TreeAdjList<NODE, EDGE>
        where EDGE : IEdge<NODE>
    {
        Dictionary<NODE, EDGE> _left = new Dictionary<NODE, EDGE>();
        Dictionary<NODE, EDGE> _right = new Dictionary<NODE, EDGE>();

        public BinaryTreeAdjList()
            : base()
        {
        }

        public NODE Left(NODE n)
        {
            if (_left.TryGetValue(n, out EDGE e))
                return e.To;
            else
                return default(NODE);
        }

        public NODE Right(NODE n)
        {
            if (_right.TryGetValue(n, out EDGE e))
                return e.To;
            else
                return default(NODE);
        }

        public override EDGE AddEdge(EDGE e)
        {
            if (this.Successors(e.From).Count() > 2)
                throw new Exception("Too many children being added to a binary tree node.");
            base.AddEdge(e);
            if (!this._left.TryGetValue(e.From, out EDGE current_left))
            {
                this._left[e.From] = e;
                return e;
            }
            if (!this._right.TryGetValue(e.From, out EDGE current_right))
            {
                this._right[e.From] = e;
                return e;
            }
            throw new Exception("Faulty logic--AddEdge");
        }
    }
}
