using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Campy.Graphs
{
    /******************************************************************************
     *  Compilation:  javac EdgeWeightedDigraph.java
     *  Execution:    java EdgeWeightedDigraph digraph.txt
     *  Dependencies: Bag.java DirectedEdge.java
     *  Data files:   https://algs4.cs.princeton.edu/44st/tinyEWD.txt
     *                https://algs4.cs.princeton.edu/44st/mediumEWD.txt
     *                https://algs4.cs.princeton.edu/44st/largeEWD.txt
     *
     *  An edge-weighted digraph, implemented using adjacency lists.
     *
     ******************************************************************************/

    /**
     *  The {@code EdgeWeightedDigraph} class represents a edge-weighted
     *  digraph of vertices named 0 through <em>V</em> - 1, where each
     *  directed edge is of type {@link DirectedEdge} and has a real-valued weight.
     *  It supports the following two primary operations: add a directed edge
     *  to the digraph and iterate over all of edges incident from a given vertex.
     *  It also provides
     *  methods for returning the number of vertices <em>V</em> and the number
     *  of edges <em>E</em>. Parallel edges and self-loops are permitted.
     *  <p>
     *  This implementation uses an adjacency-lists representation, which 
     *  is a vertex-indexed array of {@link Bag} objects.
     *  All operations take constant time (in the worst case) except
     *  iterating over the edges incident from a given vertex, which takes
     *  time proportional to the number of such edges.
     *  <p>
     *  For additional documentation,
     *  see <a href="https://algs4.cs.princeton.edu/44sp">Section 4.4</a> of
     *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
     *
     *  @author Robert Sedgewick
     *  @author Kevin Wayne
     */
    public class EdgeWeightedDigraph : GraphAdjList<int, DirectedEdge<int>>
    {
        private static string NEWLINE = Environment.NewLine;

        /**
         * Initializes an empty edge-weighted digraph with {@code V} vertices and 0 edges.
         *
         * @param  V the number of vertices
         * @throws Exception if {@code V < 0}
         */
        public EdgeWeightedDigraph(int x)
        {
            if (x < 0) throw new Exception("Number of vertices in a Digraph must be nonnegative");
           
            // KED note: the graph is "empty" in so far as there are no edges. But, it's not really "empty" as Sedgewick/Wayne
            // misleadingly say--it contains "V" number of nodes, numbered from zero.
            for (int v = 0; v < x; v++)
            {
                this.AddVertex(v);
            }
        }

        /**
         * Initializes a random edge-weighted digraph with {@code V} vertices and <em>E</em> edges.
         *
         * @param  V the number of vertices
         * @param  E the number of edges
         * @throws Exception if {@code V < 0}
         * @throws Exception if {@code E < 0}
         */
        public EdgeWeightedDigraph(int n, int e)
        {
            if (e < 0) throw new Exception("Number of edges in a Digraph must be nonnegative");
            for (int i = 0; i < e; i++)
            {
                var r = new Random(n);
                int v = r.Next(n);
                int w = r.Next(n);
                double weight = 0.01 * r.Next(100);
                DirectedEdge<int> edge = new DirectedEdge<int>(v, w, weight);
                addEdge(edge);
            }
        }

        /**  
         * Initializes an edge-weighted digraph from the specified input stream.
         * The format is the number of vertices <em>V</em>,
         * followed by the number of edges <em>E</em>,
         * followed by <em>E</em> pairs of vertices and edge weights,
         * with each entry separated by whitespace.
         *
         * @param  in the input stream
         * @throws IllegalArgumentException if the endpoints of any edge are not in prescribed range
         * @throws IllegalArgumentException if the number of vertices or edges is negative
         */
        public EdgeWeightedDigraph(string content)
        {
            try
            {
                string[] input = content.Split(new char[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

                int current = 0;

                int vertex_count = Int32.Parse(input[current++]);
                if (vertex_count < 0)
                    throw new Exception("number of vertices in a Digraph must be nonnegative");

                for (int v = 0; v < vertex_count; v++)
                {
                    this.AddVertex(v);
                }
                int edge_count = Int32.Parse(input[current++]);
                if (edge_count < 0)
                    throw new Exception("number of edges in a Digraph must be nonnegative");
                for (int i = 0; i < edge_count; i++)
                {
                    int v = Int32.Parse(input[current++]);
                    int w = Int32.Parse(input[current++]);
                    double weight = Double.Parse(input[current++]);
                    addEdge(new DirectedEdge<int>(v, w, weight));
                }
            }
            catch (Exception e)
            {
                throw new Exception("invalid input format in Digraph constructor", e);
            }
        }

        /**
         * Initializes a new edge-weighted digraph that is a deep copy of {@code G}.
         *
         * @param  G the edge-weighted digraph to copy
         */
        public EdgeWeightedDigraph(EdgeWeightedDigraph G)
        {
            foreach (var n in G.Vertices)
            {
                this.AddVertex(n);
            }
            foreach (var e in G.Edges)
            {
                this.AddEdge(e);
            }
        }

        /**
         * Returns the number of vertices in this edge-weighted digraph.
         *
         * @return the number of vertices in this edge-weighted digraph
         */
        public int V
        {
            get { return this.Vertices.Count(); }
        }

        /**
         * Returns the number of edges in this edge-weighted digraph.
         *
         * @return the number of edges in this edge-weighted digraph
         */
        public int E
        {
            get { return this.Edges.Count(); }
        }

        // throw an Exception unless {@code 0 <= v < V}
        private void validateVertex(int v)
        {
            if (v < 0 || v >= V)
                throw new Exception("vertex " + v + " is not between 0 and " + (V - 1));
        }

        /**
         * Adds the directed edge {@code e} to this edge-weighted digraph.
         *
         * @param  e the edge
         * @throws Exception unless endpoints of edge are between {@code 0}
         *         and {@code V-1}
         */
        public void addEdge(DirectedEdge<int> e)
        {
            int v = e.From;
            int w = e.To;
            validateVertex(v);
            validateVertex(w);
            base.AddEdge(e);
        }


        /**
         * Returns the directed edges incident from vertex {@code v}.
         *
         * @param  v the vertex
         * @return the directed edges incident from vertex {@code v} as an Iterable
         * @throws IllegalArgumentException unless {@code 0 <= v < V}
         */
        public IEnumerable<DirectedEdge<int>> adj(int v)
        {
            validateVertex(v);
            return this.SuccessorEdges(v);
        }

        /**
         * Returns the number of directed edges incident from vertex {@code v}.
         * This is known as the <em>outdegree</em> of vertex {@code v}.
         *
         * @param  v the vertex
         * @return the outdegree of vertex {@code v}
         * @throws Exception unless {@code 0 <= v < V}
         */
        public int outdegree(int v)
        {
            validateVertex(v);
            return this.Successors(v).Count();
        }

        /**
         * Returns the number of directed edges incident to vertex {@code v}.
         * This is known as the <em>indegree</em> of vertex {@code v}.
         *
         * @param  v the vertex
         * @return the indegree of vertex {@code v}
         * @throws Exception unless {@code 0 <= v < V}
         */
        public int indegree(int v)
        {
            validateVertex(v);
            return this.Predecessors(v).Count();
        }

        /**
         * Returns all directed edges in this edge-weighted digraph.
         * To iterate over the edges in this edge-weighted digraph, use foreach notation:
         * {@code for (DirectedEdge e : G.edges())}.
         *
         * @return all edges in this edge-weighted digraph, as an iterable
         */
        public IEnumerable<DirectedEdge<int>> edges()
        {
            return base.Edges;
        }

        /**
         * Returns a string representation of this edge-weighted digraph.
         *
         * @return the number of vertices <em>V</em>, followed by the number of edges <em>E</em>,
         *         followed by the <em>V</em> adjacency lists of edges
         */
        public String ToString()
        {
            StringBuilder s = new StringBuilder();
            s.Append(V + " " + E + NEWLINE);
            for (int v = 0; v < V; v++)
            {
                s.Append(v + ": ");
                foreach (DirectedEdge<int> e in adj(v))
                {
                    s.Append(e + "  ");
                }
                s.Append(NEWLINE);
            }
            return s.ToString();
        }

        /**
         * Unit tests the {@code EdgeWeightedDigraph} data type.
         *
         * @param args the command-line arguments
         */
        public static void test()
        {
        }
    }
}
