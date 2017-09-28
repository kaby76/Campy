
using System;
using System.Linq;

namespace Campy.Graphs
{
    public class CompressedAdjacencyList<BaseType>
    {
        FiniteTotalOrderSet<BaseType> finite_total_order;
        int index_successors_last;
        int index_successors_size;
        int index_predecessors_last;
        int index_predecessors_size;
        int data_successors_last;
        int data_successors_size;
        int data_predecessors_last;
        int data_predecessors_size;
        int[] index_successors;
        int[] data_successors;
        int[] index_predecessors;
        int[] data_predecessors;

        public CompressedAdjacencyList()
        {
            finite_total_order = new FiniteTotalOrderSet<BaseType>();
            index_successors_size = 10;
            data_successors_size = 10;
            index_predecessors_size = 10;
            data_predecessors_size = 10;
            index_successors = new int[index_successors_size];
            data_successors = new int[data_successors_size];
            index_predecessors = new int[index_predecessors_size];
            data_predecessors = new int[data_predecessors_size];
        }

        public void SetNameSpace(FiniteTotalOrderSet<BaseType> o)
        {
            finite_total_order = o;
            // Index is alway valid for node# and node# + 1.
            index_successors_size = finite_total_order.Count() + 1;
            data_successors_size = finite_total_order.Count() + 1;
            // Initially, every index is zero, so there are no edges.
            index_successors = new int[index_successors_size];
            data_successors = new int[data_successors_size];
            index_predecessors_size = finite_total_order.Count() + 1;
            data_predecessors_size = finite_total_order.Count() + 1;
            index_predecessors = new int[index_predecessors_size];
            data_predecessors = new int[data_predecessors_size];
        }

        public void Construct(FiniteTotalOrderSet<BaseType> fto)
        {
            finite_total_order = fto;
            index_successors_size = finite_total_order.Count() + 1;
            data_successors_size = finite_total_order.Count() + 1;
            index_successors = new int[index_successors_size];
            data_successors = new int[data_successors_size];
            index_predecessors_size = finite_total_order.Count() + 1;
            data_predecessors_size = finite_total_order.Count() + 1;
            index_predecessors = new int[index_predecessors_size];
            data_predecessors = new int[data_predecessors_size];
        }

        public void AddName(BaseType v)
        {
            finite_total_order.Add(v);
        }

        public int FindName(BaseType v)
        {
            return finite_total_order.BijectFromBasetype(v);
        }

        public int Add(BaseType r, BaseType c)
        {
            // Add (r,c) to the adjacency list representation of 2D table.
            // Note: r and c are zero based.

            // Find r and c in finite total order.
            int fto_r = finite_total_order.BijectFromBasetype(r);
            int fto_c = finite_total_order.BijectFromBasetype(c);

            // Expand for new entry in index_successors table, if required.
            {
                if (index_successors_last <= fto_c || index_successors_size - 1 <= fto_c)
                {
                    if (index_successors_size - 1 <= fto_c)
                    {
                        while (index_successors_size - 1 <= fto_c)
                            index_successors_size *= 2;
                        Array.Resize(ref index_successors, index_successors_size);
                    }
                    // index_successors_last must be 1 more than the last row used.
                    if (index_successors_last <= fto_c)
                    {
                        // Copy current last index_successors value to expanded portion.
                        for (int i = index_successors_last; i <= fto_c; ++i)
                            index_successors[i + 1] = index_successors[i];
                        index_successors_last = fto_c + 1;
                    }
                }
                if (index_successors_last <= fto_r || index_successors_size - 1 <= fto_r)
                {
                    if (index_successors_size - 1 <= fto_r)
                    {
                        while (index_successors_size - 1 <= fto_r)
                            index_successors_size *= 2;
                        Array.Resize(ref index_successors, index_successors_size);
                    }
                    // index_successors_last must be 1 more than the last row used.
                    if (index_successors_last <= fto_r)
                    {
                        // Copy current last index_successors value to expanded portion.
                        for (int i = index_successors_last; i <= fto_r; ++i)
                            index_successors[i + 1] = index_successors[i];
                        index_successors_last = fto_r + 1;
                    }
                }
            }
            // Expand for new entry in index_predecessors table, if required.
            {
                if (index_predecessors_last <= fto_c || index_predecessors_size - 1 <= fto_c)
                {
                    if (index_predecessors_size - 1 <= fto_c)
                    {
                        while (index_predecessors_size - 1 <= fto_c)
                            index_predecessors_size *= 2;
                        Array.Resize(ref index_predecessors, index_predecessors_size);
                    }
                    // index_predecessors_last must be 1 more than the last col used.
                    if (index_predecessors_last <= fto_c)
                    {
                        // Copy current last index_predecessors value to expanded portion.
                        for (int i = index_predecessors_last; i <= fto_c; ++i)
                            index_predecessors[i + 1] = index_predecessors[i];
                        index_predecessors_last = fto_c + 1;
                    }
                }
                if (index_predecessors_last <= fto_r || index_predecessors_size - 1 <= fto_r)
                {
                    if (index_predecessors_size - 1 <= fto_r)
                    {
                        while (index_predecessors_size - 1 <= fto_r)
                            index_predecessors_size *= 2;
                        Array.Resize(ref index_predecessors, index_predecessors_size);
                    }
                    // index_predecessors_last must be 1 more than the last col used.
                    if (index_predecessors_last <= fto_r)
                    {
                        // Copy current last index_predecessors value to expanded portion.
                        for (int i = index_predecessors_last; i <= fto_r; ++i)
                            index_predecessors[i + 1] = index_predecessors[i];
                        index_predecessors_last = fto_r + 1;
                    }
                }
            }
            int return_value = default(int);

            {
                int ir = index_successors[fto_r];
                int irp1 = index_successors[fto_r + 1];
                bool do_insert = true;
                if (data_successors_last < irp1)
                    data_successors_last = irp1;

                if (data_successors_size <= data_successors_last + 1)
                {
                    data_successors_size *= 2;
                    Array.Resize(ref data_successors, data_successors_size);
                }

                for (; ir < irp1; ++ir)
                {
                    if (data_successors[ir] == fto_c)
                    {
                        return_value = ir;
                        do_insert = false;
                        break;
                    }
                }

                if (do_insert)
                {
                    // insert (r,c).

                    // Shift data by 1 element, starting from irp1.
                    for (int i = index_successors[index_successors_last] - 1; i >= irp1; --i)
                    {
                        data_successors[i + 1] = data_successors[i];
                    }
                    // Adjust index_successors for all rows after r.
                    for (int i = fto_r + 1; i <= index_successors_last; ++i)
                    {
                        index_successors[i] += 1;
                    }
                    // Insert c.
                    data_successors[irp1] = fto_c;
                    // Adjust data_successors_last if needed.
                    if (data_successors_last < index_successors[index_successors_last])
                    {
                        data_successors_last = index_successors[index_successors_last];
                    }
                    return_value = irp1;
                }
            }

            {
                int ir = index_predecessors[fto_c];
                int irp1 = index_predecessors[fto_c + 1];
                bool do_insert = true;
                if (data_predecessors_last < irp1)
                    data_predecessors_last = irp1;

                if (data_predecessors_size <= data_predecessors_last + 1)
                {
                    data_predecessors_size *= 2;
                    Array.Resize(ref data_predecessors, data_predecessors_size);
                }

                for (; ir < irp1; ++ir)
                {
                    if (data_predecessors[ir] == fto_r)
                    {
                        do_insert = false;
                        break;
                    }
                }

                if (do_insert)
                {
                    // insert (r,c).

                    // Shift data by 1 element, starting from irp1.
                    for (int i = index_predecessors[index_predecessors_last] - 1; i >= irp1; --i)
                    {
                        data_predecessors[i + 1] = data_predecessors[i];
                    }
                    // Adjust index_successors for all rows after r.
                    for (int i = fto_c + 1; i <= index_predecessors_last; ++i)
                    {
                        index_predecessors[i] += 1;
                    }
                    // Insert r.
                    data_predecessors[irp1] = fto_r;
                    // Adjust data_successors_last if needed.
                    if (data_predecessors_last < index_predecessors[index_predecessors_last])
                    {
                        data_predecessors_last = index_predecessors[index_predecessors_last];
                    }
                }
            }

            return return_value;
        }

        public void Shrink()
        {
            Array.Resize(ref index_successors, index_successors_last + 1);
            index_successors_size = index_successors_last + 1;
            Array.Resize(ref data_successors, data_successors_last);
            data_successors_size = data_successors_last;
            Array.Resize(ref index_predecessors, index_predecessors_last + 1);
            index_predecessors_size = index_predecessors_last + 1;
            Array.Resize(ref data_predecessors, data_predecessors_last);
            data_predecessors_size = data_predecessors_last;
        }

        public int[] IndexSuccessors
        {
            get { return index_successors; }
        }

        public int[] DataSuccessors
        {
            get { return data_successors; }
        }

        public int[] IndexPredecessors
        {
            get { return index_predecessors; }
        }

        public int[] DataPredecessors
        {
            get { return data_predecessors; }
        }
    }
}
