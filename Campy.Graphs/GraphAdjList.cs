using System;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    public class GraphAdjList<NODE, EDGE> : IGraph<NODE, EDGE>
        where EDGE : IEdge<NODE>
    {
        public Dictionary<NODE, NODE> VertexSpace = new Dictionary<NODE, NODE>();
        public MultiMap<NODE, EDGE> ForwardEdgeSpace = new MultiMap<NODE, EDGE>();
        public MultiMap<NODE, EDGE> ReverseEdgeSpace = new MultiMap<NODE, EDGE>();

        class VertexEnumerator : IEnumerable<NODE>
        {
            Dictionary<NODE, NODE> VertexSpace;

            public VertexEnumerator(Dictionary<NODE, NODE> vs)
            {
                VertexSpace = vs;
            }

            public IEnumerator<NODE> GetEnumerator()
            {
                foreach (var key in VertexSpace.Keys)
                {
                    yield return key;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NODE> Vertices
        {
            get
            {
                return new VertexEnumerator(VertexSpace);
            }
        }

        public class EdgeEnumerator : IEnumerable<EDGE>
        {
            MultiMap<NODE, EDGE> EdgeSpace;

            public EdgeEnumerator(MultiMap<NODE, EDGE> es)
            {
                EdgeSpace = es;
            }

            public IEnumerator<EDGE> GetEnumerator()
            {
                foreach (var t in EdgeSpace)
                {
                    var l = t.Value;
                    foreach (var e in l)
                    {
                        yield return e;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<EDGE> Edges
        {
            get
            {
                return new EdgeEnumerator(ForwardEdgeSpace);
            }
        }

        virtual public NODE AddVertex(NODE v)
        {
            VertexSpace[v] = v;
            return v;
        }

        virtual public void DeleteVertex(NODE v)
        {
        }

        virtual public EDGE AddEdge(EDGE e)
        {
            var vf = AddVertex(e.From);
            var vt = AddVertex(e.To);
            ForwardEdgeSpace.Add(e.From, e);
            ReverseEdgeSpace.Add(e.To, e);
            return e;
        }

        virtual public void DeleteEdge(EDGE e)
        {
            var vf = e.From;
            var vt = e.To;
            var done = ForwardEdgeSpace.TryGetValue(vf, out List<EDGE> list);
            if (done)
            {
                for (int i = 0; i < list.Count; ++i)
                {
                    if (list[i].From.Equals(vf) && list[i].To.Equals(vt))
                    {
                        list.RemoveAt(i);
                        break;
                    }
                }
            }
            done = ReverseEdgeSpace.TryGetValue(vf, out List<EDGE> listr);
            if (done)
            {
                for (int i = 0; i < list.Count; ++i)
                {
                    if (list[i].From.Equals(vt) && list[i].To.Equals(vf))
                    {
                        list.RemoveAt(i);
                        break;
                    }
                }
            }
        }

        public GraphAdjList()
        {
        }

        class PredecessorEnumerator : IEnumerable<NODE>
        {
            private GraphAdjList<NODE, EDGE> graph;
            private NODE node;

            public PredecessorEnumerator(GraphAdjList<NODE, EDGE> g, NODE n)
            {
                graph = g;
                node = n;
            }

            public IEnumerator<NODE> GetEnumerator()
            {
                if (graph.ReverseEdgeSpace.TryGetValue(node, out List<EDGE> list))
                {
                    foreach (EDGE e in list)
                    {
                        yield return e.From;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NODE> Predecessors(NODE n)
        {
            return new PredecessorEnumerator(this, n);
        }

        class PredecessorEdgeEnumerator : IEnumerable<EDGE>
        {
            private GraphAdjList<NODE, EDGE> graph;
            private NODE node;

            public PredecessorEdgeEnumerator(GraphAdjList<NODE, EDGE> g, NODE n)
            {
                graph = g;
                node = n;
            }

            public IEnumerator<EDGE> GetEnumerator()
            {
                if (graph.ReverseEdgeSpace.TryGetValue(node, out List<EDGE> list))
                {
                    foreach (EDGE e in list)
                    {
                        yield return e;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<EDGE> PredecessorEdges(NODE n)
        {
            return new PredecessorEdgeEnumerator(this, n);
        }

        class ReversePredecessorEnumerator : IEnumerable<NODE>
        {
            GraphAdjList<NODE, EDGE> graph;
            NODE node;

            public ReversePredecessorEnumerator(GraphAdjList<NODE, EDGE> g, NODE n)
            {
                graph = g;
                node = n;
            }

            public IEnumerator<NODE> GetEnumerator()
            {
                if (graph.ReverseEdgeSpace.TryGetValue(node, out List<EDGE> list))
                {
                    list.Reverse();
                    foreach (var e in list)
                    {
                        yield return e.From;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NODE> ReversePredecessors(NODE n)
        {
            return new ReversePredecessorEnumerator(this, n);
        }

        public IEnumerable<NODE> PredecessorNodes(NODE n)
        {
            return new PredecessorEnumerator(this, n);
        }

        class SuccessorEnumerator : IEnumerable<NODE>
        {
            private GraphAdjList<NODE, EDGE> graph;
            private NODE node;

            public SuccessorEnumerator(GraphAdjList<NODE, EDGE> g, NODE n)
            {
                graph = g;
                node = n;
            }

            public IEnumerator<NODE> GetEnumerator()
            {
                if (graph.ForwardEdgeSpace.TryGetValue(node, out List<EDGE> list))
                {
                    foreach (EDGE e in list)
                    {
                        yield return e.To;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NODE> Successors(NODE n)
        {
            return new SuccessorEnumerator(this, n);
        }

        public IEnumerable<NODE> SuccessorNodes(NODE n)
        {
            return new SuccessorEnumerator(this, n);
        }

        class SuccessorEdgeEnumerator : IEnumerable<EDGE>
        {
            private GraphAdjList<NODE, EDGE> graph;
            private NODE node;

            public SuccessorEdgeEnumerator(GraphAdjList<NODE, EDGE> g, NODE n)
            {
                graph = g;
                node = n;
            }

            public IEnumerator<EDGE> GetEnumerator()
            {
                if (graph.ForwardEdgeSpace.TryGetValue(node, out List<EDGE> list))
                {
                    foreach (EDGE e in list)
                    {
                        yield return e;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<EDGE> SuccessorEdges(NODE n)
        {
            return new SuccessorEdgeEnumerator(this, n);
        }

        public class ReverseSuccessorEnumerator : IEnumerable<NODE>
        {
            GraphAdjList<NODE, EDGE> graph;
            NODE node;

            public ReverseSuccessorEnumerator(GraphAdjList<NODE, EDGE> g, NODE n)
            {
                graph = g;
                node = n;
            }

            public IEnumerator<NODE> GetEnumerator()
            {
                if (graph.ForwardEdgeSpace.TryGetValue(node, out List<EDGE> list))
                {
                    list.Reverse();
                    foreach (EDGE e in list)
                    {
                        yield return e.To;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NODE> ReverseSuccessors(NODE n)
        {
            return new ReverseSuccessorEnumerator(this, n);
        }

        public bool IsLeaf(NODE name)
        {
            if (this.ForwardEdgeSpace.TryGetValue(name, out List<EDGE> list))
                return false;
            else
                return true;
        }
    }
}
