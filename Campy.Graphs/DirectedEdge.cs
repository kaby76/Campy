using System;

namespace Campy.Graphs
{
    /******************************************************************************
     *  Compilation:  javac DirectedEdge.java
     *  Execution:    java DirectedEdge
     *  Dependencies: StdOut.java
     *
     *  Immutable weighted directed edge.
     *
     ******************************************************************************/
    /**
     *  The {@code DirectedEdge} class represents a weighted edge in an 
     *  {@link EdgeWeightedDigraph}. Each edge consists of two integers
     *  (naming the two vertices) and a real-value weight. The data type
     *  provides methods for accessing the two endpoints of the directed edge and
     *  the weight.
     *  <p>
     *  For additional documentation, see <a href="https://algs4.cs.princeton.edu/44sp">Section 4.4</a> of
     *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
     *
     *  @author Robert Sedgewick
     *  @author Kevin Wayne
     */

    public class DirectedEdge<NODE> : IEdge<NODE>
    {
        private double weight;
        private NODE _from;
        private NODE _to;

        /**
         * Initializes a directed edge from vertex {@code v} to vertex {@code w} with
         * the given {@code weight}.
         * @param v the tail vertex
         * @param w the head vertex
         * @param weight the weight of the directed edge
         */
        public DirectedEdge(NODE v, NODE w, double weight = 0)
        {
            if (Double.IsNaN(weight)) throw new Exception("Weight is NaN");
            this._from = v;
            this._to = w;
            this.weight = weight;
        }

        /**
         * Returns the tail vertex of the directed edge.
         * @return the tail vertex of the directed edge
         */
        public NODE From
        {
            get { return _from; }
            set {_from = value; }
        }


        /**
         * Returns the head vertex of the directed edge.
         * @return the head vertex of the directed edge
         */
        public NODE To
        {
            get { return _to; }
            set { _to = value; }
        }

        /**
         * Returns the weight of the directed edge.
         * @return the weight of the directed edge
         */
        public double Weight()
        {
            return weight;
        }

        /**
         * Returns a string representation of the directed edge.
         * @return a string representation of the directed edge
         */
        public int CompareTo(IEdge<NODE> other)
        {
            throw new NotImplementedException();
        }

        public override string ToString()
        {
            return _from + "->" + _to + " " + String.Format("%5.2f", weight);
        }

        /**
         * Unit tests the {@code DirectedEdge} data type.
         *
         * @param args the command-line arguments
         */
        public static void test()
        {
            DirectedEdge<int> e = new DirectedEdge<int>(12, 34, 5.67);
            System.Console.WriteLine(e);
        }
    }
}
