using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace Campy.Graphs
{
    public class GraphAdjList<NAME> : IGraph<NAME>
    {
        public FiniteTotalOrderSet<NAME> NameSpace = new FiniteTotalOrderSet<NAME>();
        public GraphAdjListVertex<NAME>[] VertexSpace = new GraphAdjListVertex<NAME>[10];
        public GraphAdjListEdge<NAME>[] EdgeSpace = new GraphAdjListEdge<NAME>[10];
        public CompressedAdjacencyList<NAME> adj = new CompressedAdjacencyList<NAME>();
        protected Type NODE;
        protected Type EDGE;

        class VertexEnumerator : IEnumerable<NAME>
        {
            GraphAdjListVertex<NAME>[] VertexSpace;

            public VertexEnumerator(GraphAdjListVertex<NAME>[] vs)
            {
                VertexSpace = vs;
            }

            public IEnumerator<NAME> GetEnumerator()
            {
                for (int i = 0; i < VertexSpace.Length; ++i)
                {
                    if (VertexSpace[i] != null)
                        yield return VertexSpace[i].Name;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NAME> Vertices
        {
            get
            {
                return new VertexEnumerator(VertexSpace);
            }
        }

        public class EdgeEnumerator : IEnumerable<GraphAdjListEdge<NAME>>
        {
            GraphAdjListEdge<NAME>[] EdgeSpace;

            public EdgeEnumerator(GraphAdjListEdge<NAME>[] es)
            {
                EdgeSpace = es;
            }

            public IEnumerator<GraphAdjListEdge<NAME>> GetEnumerator()
            {
                for (int i = 0; i < EdgeSpace.Length; ++i)
                {
                    if (EdgeSpace[i] != null)
                        yield return EdgeSpace[i];
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<IEdge<NAME>> Edges
        {
            get
            {
                return new EdgeEnumerator(EdgeSpace);
            }
        }

        virtual protected GraphAdjListVertex<NAME> CreateVertex()
        {
            GraphAdjListVertex<NAME> vv = (GraphAdjListVertex<NAME>)Activator.CreateInstance(NODE, true);
            return vv;
        }

        virtual public GraphAdjListVertex<NAME> AddVertex(NAME v)
        {
            GraphAdjListVertex<NAME> vv = null;

            // NB: This code is very efficient if the name space
            // is integer, and has been preconstructed. Otherwise,
            // it will truly suck in speed.

            // Add name.
            NameSpace.Add(v);

            // Find bijection v into int domain.
            int iv = NameSpace.BijectFromBasetype(v);

            // Find node from int domain.
            if (iv >= VertexSpace.Length)
            {
                Array.Resize(ref VertexSpace, VertexSpace.Length * 2);
            }
            if (VertexSpace[iv] == null)
            {
                vv = (GraphAdjListVertex<NAME>)CreateVertex();
                vv.Name = v;
                vv._Graph = this;
                VertexSpace[iv] = vv;
            }
            else
                vv = VertexSpace[iv];
            return vv;
        }

        virtual public void DeleteVertex(GraphAdjListVertex<NAME> vertex)
        {
        }

        virtual protected GraphAdjListEdge<NAME> CreateEdge()
        {
            GraphAdjListEdge<NAME> e = (GraphAdjListEdge<NAME>)Activator.CreateInstance(EDGE, true);
            return e;
        }

        virtual public GraphAdjListEdge<NAME> AddEdge(NAME f, NAME t)
        {
            GraphAdjListVertex<NAME> vf = AddVertex(f);
            GraphAdjListVertex<NAME> vt = AddVertex(t);
            // Create adjacency table entry for (f, t).
            int j = adj.Add(f, t);
            // Create EDGE with from/to.
            if (j >= EdgeSpace.Length)
            {
                Array.Resize(ref EdgeSpace, EdgeSpace.Length * 2);
            }
            if (EdgeSpace[j] == null)
            {
                GraphAdjListEdge<NAME> edge = CreateEdge();
                edge.to = vt;
                edge.from = vf;
                EdgeSpace[j] = edge;
                return edge;
            }
            else
                return EdgeSpace[j];
        }

        virtual public void DeleteEdge(NAME f, NAME t)
        {
        }

        public void SetNameSpace(IEnumerable<NAME> ns)
        {
            NameSpace.OrderedRelationship(ns);
            adj.Construct(NameSpace);
        }

        public void Optimize()
        {
            adj.Shrink();
        }

        public GraphAdjList()
        {
            NODE = typeof(GraphAdjListVertex<NAME>);
            EDGE = typeof(GraphAdjListEdge<NAME>);
        }

        class PredecessorEnumerator : IEnumerable<NAME>
        {
            GraphAdjList<NAME> graph;
            NAME name;

            public PredecessorEnumerator(GraphAdjList<NAME> g, NAME n)
            {
                graph = g;
                name = n;
            }

            public IEnumerator<NAME> GetEnumerator()
            {
                int[] index = graph.adj.IndexPredecessors;
                int[] data = graph.adj.DataPredecessors;
                int n = graph.NameSpace.BijectFromBasetype(name);
                GraphAdjListVertex<NAME> node = graph.VertexSpace[n];
                for (int i = index[n]; i < index[n + 1]; ++i)
                {
                    int d = data[i];
                    NAME c = graph.VertexSpace[d].Name;
                    yield return c;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NAME> Predecessors(NAME n)
        {
            return new PredecessorEnumerator(this, n);
        }

        class PredecessorEdgeEnumerator : IEnumerable<IEdge<NAME>>
        {
            GraphAdjList<NAME> graph;
            NAME name;

            public PredecessorEdgeEnumerator(GraphAdjList<NAME> g, NAME n)
            {
                graph = g;
                name = n;
            }

            public IEnumerator<IEdge<NAME>> GetEnumerator()
            {
                int[] index = graph.adj.IndexPredecessors;
                int[] data = graph.adj.DataPredecessors;
                int n = graph.NameSpace.BijectFromBasetype(name);
                GraphAdjListVertex<NAME> node = graph.VertexSpace[n];
                for (int i = index[n]; i < index[n + 1]; ++i)
                {
                    int d = data[i];
                    var c = graph.VertexSpace[d];
                    yield return new GraphAdjListEdge<NAME>(c, node);
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }


        public IEnumerable<IEdge<NAME>> PredecessorEdges(NAME n)
        {
            return new PredecessorEdgeEnumerator(this, n);
        }

        class ReversePredecessorEnumerator : IEnumerable<NAME>
        {
            GraphAdjList<NAME> graph;
            NAME name;

            public ReversePredecessorEnumerator(GraphAdjList<NAME> g, NAME n)
            {
                graph = g;
                name = n;
            }

            public IEnumerator<NAME> GetEnumerator()
            {
                int[] index = graph.adj.IndexPredecessors;
                int[] data = graph.adj.DataPredecessors;
                int n = graph.NameSpace.BijectFromBasetype(name);
                GraphAdjListVertex<NAME> node = graph.VertexSpace[n];
                for (int i = index[n + 1] - 1; i >= index[n]; --i)
                {
                    int d = data[i];
                    NAME c = graph.VertexSpace[d].Name;
                    yield return c;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NAME> ReversePredecessors(NAME n)
        {
            return new ReversePredecessorEnumerator(this, n);
        }

        class SuccessorEnumerator : IEnumerable<NAME>
        {
            GraphAdjList<NAME> graph;
            NAME name;

            public SuccessorEnumerator(GraphAdjList<NAME> g, NAME n)
            {
                graph = g;
                name = n;
            }

            public IEnumerator<NAME> GetEnumerator()
            {
                int[] index = graph.adj.IndexSuccessors;
                int[] data = graph.adj.DataSuccessors;
                int n = graph.NameSpace.BijectFromBasetype(name);
                GraphAdjListVertex<NAME> node = graph.VertexSpace[n];
                for (int i = index[n]; i < index[n + 1]; ++i)
                {
                    int d = data[i];
                    NAME c = graph.VertexSpace[d].Name;
                    yield return c;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NAME> Successors(NAME n)
        {
            return new SuccessorEnumerator(this, n);
        }

        class SuccessorEdgeEnumerator : IEnumerable<IEdge<NAME>>
        {
            GraphAdjList<NAME> graph;
            NAME name;

            public SuccessorEdgeEnumerator(GraphAdjList<NAME> g, NAME n)
            {
                graph = g;
                name = n;
            }

            public IEnumerator<IEdge<NAME>> GetEnumerator()
            {
                int[] index = graph.adj.IndexSuccessors;
                int[] data = graph.adj.DataSuccessors;
                int n = graph.NameSpace.BijectFromBasetype(name);
                GraphAdjListVertex<NAME> node = graph.VertexSpace[n];
                for (int i = index[n]; i < index[n + 1]; ++i)
                {
                    int d = data[i];
                    var c = graph.VertexSpace[d];
                    yield return new GraphAdjListEdge<NAME>(node, c);
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }


        public IEnumerable<IEdge<NAME>> SuccessorEdges(NAME n)
        {
            return new SuccessorEdgeEnumerator(this, n);
        }

        public class ReverseSuccessorEnumerator : IEnumerable<NAME>
        {
            GraphAdjList<NAME> graph;
            NAME name;

            public ReverseSuccessorEnumerator(GraphAdjList<NAME> g, NAME n)
            {
                graph = g;
                name = n;
            }

            public IEnumerator<NAME> GetEnumerator()
            {
                int[] index = graph.adj.IndexSuccessors;
                int[] data = graph.adj.DataSuccessors;
                int n = graph.NameSpace.BijectFromBasetype(name);
                GraphAdjListVertex<NAME> node = graph.VertexSpace[n];
                for (int i = index[n + 1] - 1; i >= index[n]; --i)
                {
                    int d = data[i];
                    NAME c = graph.VertexSpace[d].Name;
                    yield return c;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<NAME> ReverseSuccessors(NAME n)
        {
            return new ReverseSuccessorEnumerator(this, n);
        }

        public bool IsLeaf(NAME n)
        {
            return !Successors(n).Any();
        }
    }

    public class GraphAdjListVertex<NAME> : IVertex<NAME>, IComparable<GraphAdjListVertex<NAME>>
    {
        public NAME Name
        {
            get;
            set;
        }

        public int Index
        {
            get
            {
                int i = _Graph.adj.FindName(Name);
                return i;
            }
        }

        public GraphAdjList<NAME> _Graph
        {
            get;
            set;
        }

        public GraphAdjListVertex()
        {
        }

        public GraphAdjListVertex(NAME t)
        {
            this.Name = t;
        }

        public int CompareTo(IVertex<NAME> other)
        {
            throw new NotImplementedException();
        }

        override public string ToString()
        {
            return Name.ToString();
        }

        public int CompareTo(GraphAdjListVertex<NAME> other)
        {
            throw new NotImplementedException();
        }
    }

    public class GraphAdjListEdge<NAME> : IEdge<NAME>
    {
        public GraphAdjListVertex<NAME> from;

        public GraphAdjListVertex<NAME> to;

        public GraphAdjListEdge()
        {
        }

        public GraphAdjListEdge(GraphAdjListVertex<NAME> f, GraphAdjListVertex<NAME> t)
        {
            from = (GraphAdjListVertex<NAME>)f;
            to = (GraphAdjListVertex<NAME>)t;
        }

        public NAME From
        {
            get { return from.Name; }
        }

        public NAME To
        {
            get { return to.Name; }
        }

        public int CompareTo(IEdge<NAME> other)
        {
            throw new NotImplementedException();
        }

        override public string ToString()
        {
            return "(" + from.Name + ", " + to.Name + ")";
        }
    }
}
