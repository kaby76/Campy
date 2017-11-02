/******************************************************************************
 *  Compilation:  javac GabowSCC.java
 *  Execution:    java GabowSCC V E
 *  Dependencies: Digraph.java Stack.java TransitiveClosure.java StdOut.java
 *  Data files:   https://algs4.cs.princeton.edu/42digraph/tinyDG.txt
 *                https://algs4.cs.princeton.edu/42digraph/mediumDG.txt
 *                https://algs4.cs.princeton.edu/42digraph/largeDG.txt
 *
 *  Compute the strongly-connected components of a digraph using 
 *  Gabow's algorithm (aka Cheriyan-Mehlhorn algorithm).
 *
 *  Runs in O(E + V) time.
 *
 *  % java GabowSCC tinyDG.txt
 *  5 components
 *  1 
 *  0 2 3 4 5
 *  9 10 11 12
 *  6 8
 *  7 
 *
 ******************************************************************************/
/**
 *  The {@code GabowSCC} class represents a data type for 
 *  determining the strong components in a digraph.
 *  The <em>id</em> operation determines in which strong component
 *  a given vertex lies; the <em>areStronglyConnected</em> operation
 *  determines whether two vertices are in the same strong component;
 *  and the <em>count</em> operation determines the number of strong
 *  components.

 *  The <em>component identifier</em> of a component is one of the
 *  vertices in the strong component: two vertices have the same component
 *  identifier if and only if they are in the same strong component.

 *  <p>
 *  This implementation uses the Gabow's algorithm.
 *  The constructor takes time proportional to <em>V</em> + <em>E</em>
 *  (in the worst case),
 *  where <em>V</em> is the number of vertices and <em>E</em> is the number of edges.
 *  Afterwards, the <em>id</em>, <em>count</em>, and <em>areStronglyConnected</em>
 *  operations take constant time.
 *  For alternate implementations of the same API, see
 *  {@link KosarajuSharirSCC} and {@link TarjanSCC}.
 *  <p>
 *  For additional documentation,
 *  see <a href="https://algs4.cs.princeton.edu/42digraph">Section 4.2</a> of
 *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 *
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Campy.Utils;

namespace Campy.Graphs
{
    public class Gabow
    {

        private bool[] marked; // marked[v] = has v been visited?
        private int[] id; // id[v] = id of strong component containing v
        private int[] preorder; // preorder[v] = preorder of v
        private int pre; // preorder number counter
        private int count; // number of strongly-connected components
        private Stack<int> stack1;
        private Stack<int> stack2;

        /**
         * Computes the strong components of the digraph {@code G}.
         * @param G the digraph
         */
        public Gabow(Digraph G)
        {
            marked = new bool[G.V];
            stack1 = new Stack<int>();
            stack2 = new Stack<int>();
            id = new int[G.V];
            preorder = new int[G.V];
            for (int v = 0; v < G.V; v++)
                id[v] = -1;

            for (int v = 0; v < G.V; v++)
            {
                if (!marked[v]) dfs(G, v);
            }
        }

        private void dfs(Digraph G, int v)
        {
            marked[v] = true;
            preorder[v] = pre++;
            stack1.Push(v);
            stack2.Push(v);
            foreach (int w in G.adj(v))
            {
                if (!marked[w]) dfs(G, w);
                else if (id[w] == -1)
                {
                    while (preorder[stack2.Peek()] > preorder[w])
                        stack2.Pop();
                }
            }

            // found strong component containing v
            if (stack2.Peek() == v)
            {
                stack2.Pop();
                int w;
                do
                {
                    w = stack1.Pop();
                    id[w] = count;
                } while (w != v);
                count++;
            }
        }

        /**
         * Returns the number of strong components.
         * @return the number of strong components
         */
        public int Count()
        {
            return count;
        }

        /**
         * Are vertices {@code v} and {@code w} in the same strong component?
         * @param  v one vertex
         * @param  w the other vertex
         * @return {@code true} if vertices {@code v} and {@code w} are in the same
         *         strong component, and {@code false} otherwise
         * @throws IllegalArgumentException unless {@code 0 <= v < V}
         * @throws IllegalArgumentException unless {@code 0 <= w < V}
         */
        public bool stronglyConnected(int v, int w)
        {
            validateVertex(v);
            validateVertex(w);
            return id[v] == id[w];
        }

        /**
         * Returns the component id of the strong component containing vertex {@code v}.
         * @param  v the vertex
         * @return the component id of the strong component containing vertex {@code v}
         * @throws IllegalArgumentException unless {@code 0 <= v < V}
         */
        public int Id(int v)
        {
            validateVertex(v);
            return id[v];
        }

        // does the id[] array contain the strongly connected components?
        private bool check(Digraph G)
        {
            TransitiveClosure tc = new TransitiveClosure(G);
            for (int v = 0; v < G.V; v++)
            {
                for (int w = 0; w < G.V; w++)
                {
                    if (stronglyConnected(v, w) != (tc.reachable(v, w) && tc.reachable(w, v)))
                        return false;
                }
            }
            return true;
        }

        // throw an IllegalArgumentException unless {@code 0 <= v < V}
        private void validateVertex(int v)
        {
            int V = marked.Length;
            if (v < 0 || v >= V)
                throw new Exception("vertex " + v + " is not between 0 and " + (V - 1));
        }

        /**
         * Unit tests the {@code GabowSCC} data type.
         *
         * @param args the command-line arguments
         */
        public static void test()
        {
            string tiny = $@"
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

            Digraph G = new Digraph(tiny);
            Gabow scc = new Gabow(G);

            // number of connected components
            int m = scc.Count();
            System.Console.WriteLine(m + " components");

            // compute list of vertices in each strong component
            Queue<int>[] components = new Queue<int>[m];
            for (int i = 0; i < m; i++)
            {
                components[i] = new Queue<int>();
            }
            for (int v = 0; v < G.Vertices.Count(); v++)
            {
                components[scc.Id(v)].Enqueue(v);
            }

            // print results
            for (int i = 0; i < m; i++)
            {
                foreach (int v in components[i])
                {
                    System.Console.WriteLine(v + " ");
                }
                System.Console.WriteLine();
            }
        }
    }
}