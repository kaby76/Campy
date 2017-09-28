using System;
using System.Collections.Generic;

namespace Campy.Graphs
{
    public class FiniteTotalOrderSet<SetElementType> : IEnumerable<SetElementType>
    {
        List<SetElementType> elements = new List<SetElementType>();
        List<Tuple<SetElementType, SetElementType>> relation = new List<Tuple<SetElementType, SetElementType>>();

        // Notes whether to convert relationships, etc. into final
        // representation, and to perform optimizations.
        bool finalized = false;

        // Bijective functions for conversion from basetype to int and vice versa.
        List<SetElementType> int_to_basetype = new List<SetElementType>();
        Dictionary<SetElementType, int> basetype_to_int = new Dictionary<SetElementType, int>();

        // Optimizations in the case where element type is int.
        bool optimized = false;
        int start_index = 0;

        public FiniteTotalOrderSet()
        {
        }

        /// <summary>
        /// A total ordered set with relationship defined by ordered list.
        /// If the set element type is int, then it will check the list
        /// order and set it up for optimizations in the bijective functions
        /// mapping the set element into an ordinal, and vice versa.
        /// </summary>
        /// <param name="list"></param>
        public FiniteTotalOrderSet(IEnumerable<SetElementType> list)
        {
            {
                int i = 0;
                foreach (SetElementType e in list)
                {
                    int_to_basetype.Add(e);
                    basetype_to_int.Add(e, i);
                    i++;
                }
            }
            optimized = false;
            // Determine whether to optimize bijection functions:
            // if basetype is int, and the entire array is integer order,
            // then it can be optimized.
            if (typeof(SetElementType) == typeof(int))
            {
                // Crappy C# in the way.
                string str = int_to_basetype[0].ToString();
                int prev = Convert.ToInt32(str);
                for (int i = 1; i < int_to_basetype.Count; ++i)
                {
                    str = int_to_basetype[i].ToString();
                    int next = Convert.ToInt32(str);
                    if (prev > next)
                    {
                        optimized = false;
                        return;
                    }
                }
                optimized = true;
                str = int_to_basetype[0].ToString();
                start_index = Convert.ToInt32(str);
            }
            finalized = true;
        }

        public void OrderedRelationship(IEnumerable<SetElementType> list)
        {
            {
                int i = 0;
                foreach (SetElementType e in list)
                {
                    int_to_basetype.Add(e);
                    basetype_to_int.Add(e, i);
                    i++;
                }
            }
            optimized = false;
            // Determine whether to optimize bijection functions:
            // if basetype is int, and the entire array is integer order,
            // then it can be optimized.
            if (typeof(SetElementType) == typeof(int))
            {
                // Crappy C# in the way.
                string str = int_to_basetype[0].ToString();
                int prev = Convert.ToInt32(str);
                for (int i = 1; i < int_to_basetype.Count; ++i)
                {
                    str = int_to_basetype[i].ToString();
                    int next = Convert.ToInt32(str);
                    if (prev > next)
                    {
                        optimized = false;
                        return;
                    }
                }
                optimized = true;
                str = int_to_basetype[0].ToString();
                start_index = Convert.ToInt32(str);
            }
            finalized = true;
        }

        public void Add(SetElementType v)
        {
            // If optimized, then return since there's nothing to enter.
            if (optimized)
                return;

            if (int_to_basetype.Contains(v))
                return;

            int_to_basetype.Add(v);
            basetype_to_int.Add(v, basetype_to_int.Count);
            // Determine whether to optimize bijection functions:
            // if basetype is int, and the entire array is integer order,
            // then it can be optimized.
            //if (typeof(SetElementType) == typeof(int))
            //{
            //    // Crappy C# in the way.
            //    string str = int_to_basetype[0].ToString();
            //    int prev = Convert.ToInt32(str);
            //    for (int i = 1; i < int_to_basetype.Count; ++i)
            //    {
            //        str = int_to_basetype[i].ToString();
            //        int next = Convert.ToInt32(str);
            //        if (prev > next)
            //        {
            //            optimized = false;
            //            return;
            //        }
            //    }
            //    optimized = true;
            //    str = int_to_basetype[0].ToString();
            //    start_index = Convert.ToInt32(str);
            //}
        }

        public void OrderedRelation(SetElementType lhs, SetElementType rhs)
        {
            relation.Add(new Tuple<SetElementType, SetElementType>(lhs, rhs));
        }

        private void DFS(int[] depth, SetElementType e, int d)
        {
            // Set depth of e to d, then work on all "children".
            int index = basetype_to_int[e];
            depth[index] = d;
            foreach (Tuple<SetElementType, SetElementType> r in relation)
            {
                if (r.Item1.Equals(e))
                {
                    DFS(depth, r.Item2, d + 1);
                }
            }
        }

        public SetElementType BijectFromInt(int k)
        {
            if (!finalized)
            {
                finalized = true;
                // Convert ordered pairs into ordered list, making sure
                // the pairs define a total order.
                bool first = true;
                SetElementType min = default(SetElementType);
                foreach (Tuple<SetElementType, SetElementType> t in relation)
                {
                    // Find least set element.
                    if (first)
                    {
                        min = t.Item1;
                        first = false;
                    }
                    else if (t.Item2.Equals(min))
                    {
                        min = t.Item1;
                    }
                    if (!basetype_to_int.ContainsKey(t.Item1))
                        basetype_to_int.Add(t.Item1, 0);
                    if (!basetype_to_int.ContainsKey(t.Item2))
                        basetype_to_int.Add(t.Item2, 0);
                }
                // Min found.
                // Intialize basetype_to_int to an initial guess.
                {
                    int c = 0;
                    foreach (KeyValuePair<SetElementType, int> p in basetype_to_int)
                    {
                        basetype_to_int[p.Key] = c++;
                    }
                }
                // Perform BFS traversal to establish depth of each
                // element in a total order. This will be the bijective mapping
                // to int.
                int[] depth = new int[elements.Count];
                DFS(depth, min, 0);

                // Determine whether to optimize bijection functions:
                // if basetype is int, and the entire array is integer order,
                // then it can be optimized.
                if (typeof(SetElementType) == typeof(int))
                {
                    bool ok = true;
                    // Crappy C# in the way.
                    string str = int_to_basetype[0].ToString();
                    int prev = Convert.ToInt32(str);
                    for (int i = 1; i < int_to_basetype.Count; ++i)
                    {
                        str = int_to_basetype[i].ToString();
                        int next = Convert.ToInt32(str);
                        if (prev > next)
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (ok)
                    {
                        optimized = true;
                        str = int_to_basetype[0].ToString();
                        start_index = Convert.ToInt32(str);
                    }
                    else
                        optimized = false;
                }
            }

            if (!optimized)
                return int_to_basetype[k];
            else
                return (SetElementType)Convert.ChangeType((k - start_index), typeof(SetElementType));
        }

        public int BijectFromBasetype(SetElementType v)
        {
            if (!optimized)
                return basetype_to_int[v];
            else
                return (int)Convert.ChangeType(v, typeof(int)) + start_index;
        }

        public System.Collections.Generic.IEnumerator<SetElementType> GetEnumerator()
        {
            for (int i = 0; i < int_to_basetype.Count; ++i)
            {
                SetElementType b = int_to_basetype[i];
                yield return b;
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
