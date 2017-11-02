using System;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Graphs
{
    public class GraphAdjList<NODE, EDGE> : IGraph<NODE, EDGE>
        where EDGE : IEdge<NODE>
    {
        bool allow_duplicates = false;

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
        }

        virtual public void Optimize()
        {
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
                foreach (EDGE e in graph.ReverseEdgeSpace[node])
                {
                    yield return e.From;
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
                foreach (EDGE e in graph.ReverseEdgeSpace[node])
                {
                    yield return e;
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
                var x = graph.ReverseEdgeSpace[node];
                x.Reverse();
                foreach (var e in x)
                {
                    yield return e.From;
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
                foreach (EDGE e in graph.ForwardEdgeSpace[node])
                {
                    yield return e.To;
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
                foreach (EDGE e in graph.ForwardEdgeSpace[node])
                {
                    yield return e;
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
                var x = graph.ForwardEdgeSpace[node];
                x.Reverse();
                foreach (EDGE e in x)
                {
                    yield return e.To;
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
