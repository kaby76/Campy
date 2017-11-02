using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Campy.Graphs
{
    /******************************************************************************
     *  Compilation:  javac DirectedDFS.java
     *  Execution:    java DirectedDFS digraph.txt s
     *  Dependencies: Digraph.java Bag.java In.java StdOut.java
     *  Data files:   https://algs4.cs.princeton.edu/42digraph/tinyDG.txt
     *                https://algs4.cs.princeton.edu/42digraph/mediumDG.txt
     *                https://algs4.cs.princeton.edu/42digraph/largeDG.txt
     *
     *  Determine single-source or multiple-source reachability in a digraph
     *  using depth first search.
     *  Runs in O(E + V) time.
     *
     *  % java DirectedDFS tinyDG.txt 1
     *  1
     *
     *  % java DirectedDFS tinyDG.txt 2
     *  0 1 2 3 4 5
     *
     *  % java DirectedDFS tinyDG.txt 1 2 6
     *  0 1 2 3 4 5 6 8 9 10 11 12 
     *
     ******************************************************************************/
    /**
     *  The {@code DirectedDFS} class represents a data type for 
     *  determining the vertices reachable from a given source vertex <em>s</em>
     *  (or set of source vertices) in a digraph. For versions that find the paths,
     *  see {@link DepthFirstDirectedPaths} and {@link BreadthFirstDirectedPaths}.
     *  <p>
     *  This implementation uses depth-first search.
     *  The constructor takes time proportional to <em>V</em> + <em>E</em>
     *  (in the worst case),
     *  where <em>V</em> is the number of vertices and <em>E</em> is the number of edges.
     *  <p>
     *  For additional documentation,
     *  see <a href="https://algs4.cs.princeton.edu/42digraph">Section 4.2</a> of
     *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
     *
     *  @author Robert Sedgewick
     *  @author Kevin Wayne
     */
    public class DirectedDFS
    {
        private bool[] marked; // marked[v] = true if v is reachable

        // from source (or sources)
        private int count; // number of vertices reachable from s

        /**
         * Computes the vertices in digraph {@code G} that are
         * reachable from the source vertex {@code s}.
         * @param G the digraph
         * @param s the source vertex
         * @throws Exception unless {@code 0 <= s < V}
         */
        public DirectedDFS(Digraph G, int s)
        {
            marked = new bool[G.V];
            validateVertex(s);
            dfs(G, s);
        }

        /**
         * Computes the vertices in digraph {@code G} that are
         * connected to any of the source vertices {@code sources}.
         * @param G the graph
         * @param sources the source vertices
         * @throws Exception unless {@code 0 <= s < V}
         *         for each vertex {@code s} in {@code sources}
         */
        public DirectedDFS(Digraph G, IEnumerable<int> sources)
        {
            marked = new bool[G.V];
            validateVertices(sources);
            foreach (int v in sources)
            {
                if (!Marked(v)) dfs(G, v);
            }
        }

        private void dfs(Digraph G, int v)
        {
            count++;
            marked[v] = true;
            foreach (int w in G.adj(v))
            {
                if (!marked[w]) dfs(G, w);
            }
        }

        /**
         * Is there a directed path from the source vertex (or any
         * of the source vertices) and vertex {@code v}?
         * @param  v the vertex
         * @return {@code true} if there is a directed path, {@code false} otherwise
         * @throws Exception unless {@code 0 <= v < V}
         */
        public bool Marked(int v)
        {
            validateVertex(v);
            return marked[v];
        }

        /**
         * Returns the number of vertices reachable from the source vertex
         * (or source vertices).
         * @return the number of vertices reachable from the source vertex
         *   (or source vertices)
         */
        public int Count()
        {
            return count;
        }

        // throw an Exception unless {@code 0 <= v < V}
        private void validateVertex(int v)
        {
            int V = marked.Length;
            if (v < 0 || v >= V)
                throw new Exception("vertex " + v + " is not between 0 and " + (V - 1));
        }

        // throw an Exception unless {@code 0 <= v < V}
        private void validateVertices(IEnumerable<int> vertices)
        {
            if (vertices == null)
            {
                throw new Exception("argument is null");
            }
            int V = marked.Length;
            foreach (int v in vertices)
            {
                if (v < 0 || v >= V)
                {
                    throw new Exception("vertex " + v + " is not between 0 and " + (V - 1));
                }
            }
        }


        /**
         * Unit tests the {@code DirectedDFS} data type.
         *
         * @param args the command-line arguments
         */
        public static void test()
        {
            string tiny = @"
13
22
 4  2
 2  3
 3  2
 6  0
 0  1
 2  0
11 12
12  9
 9 10
 9 11
 7  9
10 12
11  4
 4  3
 3  5
 6  8
 8  6
 5  4
 0  5
 6  4
 6  9
 7  6
";

            // read in digraph from command-line argument
            Digraph G = new Digraph(tiny);
            List<int> sources = new List<int>();
            sources.Add(1);
            sources.Add(2);
            sources.Add(6);

            // multiple-source reachability
            DirectedDFS dfs = new DirectedDFS(G, sources);

            // print out vertices reachable from sources
            for (int v = 0; v < G.V; v++)
            {
                if (dfs.Marked(v)) System.Console.Write(v + " ");
            }
            System.Console.WriteLine();
        }

    }
}