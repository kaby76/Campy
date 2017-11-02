using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Campy.Utils;

namespace Campy.Graphs
{
    // Grama A, Karypis G, Kumar V, Gupta A. Introduction to Parallel
    // Computing. 2nd ed. Essex, England: Pearson Education Limited;
    // 2003.
    //
    // Rao VN, Kumar V. Parallel depth first search. Part I.
    // Implementation. International Journal of Parallel Programming.
    // 1987; 16(6):479-99.
    public class DFSParallel<T, E>
        where E : IEdge<T>
    {
        IGraph<T,E> graph;
        IEnumerable<T> Source;
        Dictionary<T, bool> Visited = new Dictionary<T, bool>();
        int NumberOfWorkers;

        public DFSParallel(IGraph<T,E> g, IEnumerable<T> s, int n)
        {
            graph = g;
            Source = s;
            NumberOfWorkers = n;
            foreach (T v in graph.Vertices)
                Visited.Add(v, false);
        }

        // Per-thread stack, contains tuple containing backtrack and to do list.
        StackQueue<Tuple<T, StackQueue<T>>>[] Stack;

        int NUMRETRY = 10;
        int CutOff = 3;
        Object obj = new Object();


        // Copy work from another thread to thread "index_successors".
        void GetWork(int index)
        {
            lock (Stack[index])
            {
                // Clean up.
                while (Stack[index].Count > 1 && Stack[index].PeekTop().Item2.Count == 0)
                    Stack[index].Pop();

                // Check if there is work.
                if (!(Stack[index].Count == 1 && Stack[index].PeekTop().Item2.Count == 0))
                    return;
            }

            bool done = false;
            int from = 0;
            for (int j = 0; j < NUMRETRY - 1; ++j)
            {
                from = (from + 1) % NumberOfWorkers;

                lock (Stack[from])
                {
                    if (Stack[from].Count > CutOff)
                    {
                        // Check if there actually is work in Stack[from].
                        // There may be a stack full of empty "todo" work lists, in which case the
                        // thread that owns the stack hasn't yet cleaned up.
                        int count = 0;
                        for (int i = 0; i < Stack[from].Count - CutOff; ++i)
                        {
                            count += Stack[from].PeekBottom(i).Item2.Count;
                        }
                        if (count <= 1)
                            continue;

                        count = count / 2;
                        if (count < 1)
                            continue;

                        //System.Console.WriteLine("Stealing " + count + " work items from " + from + " to " + index_successors);

                        // Work is available at the stack of threads.
                        // Grab "count" nodes to work on.
                        
                        // Copy stack "from" to "index_successors",
                        // and then divide the two stacks into disjoint sets of
                        // vertices.
                        StackQueue<Tuple<T, StackQueue<T>>> new_stack =
                            new StackQueue<Tuple<T, StackQueue<T>>>();
                        for (int i = 0; i < Stack[from].Count - CutOff && count > 0; ++i)
                        {
                            Tuple<T, StackQueue<T>> tf = Stack[from].PeekBottom(i);
                            T tb = tf.Item1;
                            StackQueue<T> s = tf.Item2;

                            // Make partitions.
                            StackQueue<T> work = new StackQueue<T>();
                            for ( ; count > 0 && s.Count != 0; --count)
                            {
                                T v = s.DequeueBottom(); // Side effect removing work.
                                work.Push(v);
                            }

                            Tuple<T, StackQueue<T>> tt =
                                new Tuple<T, StackQueue<T>>(tb, work);

                            new_stack.Push(tt);
                        }

                        // assign new stack.
                        Stack[index] = new_stack;

                        done = true;
                    }
                }
                if (done)
                    return;
            }
        }

        bool Terminate = false;

        bool TerminateTest()
        {
            // If goal found, terminate.
            if (Terminate)
                return true;

            // If no processors have any work, terminate.
            bool done = true;
            for (int target = 0; target < NumberOfWorkers; ++target)
            {
                lock (Stack[target])
                {
                    if (Stack[target].Count != 1)
                    {
                        done = false;
                        break;
                    }
                    if (Stack[target].PeekTop().Item2.Count != 0)
                    {
                        done = false;
                        break;
                    }
                }
            }
            if (done)
                Terminate = true;

            return Terminate;
        }

        bool SpecialContains(int index, T v)
        {
            // Look up stack and see if v is on backtrack list.
            for (int i = 0; i < Stack[index].Count; ++i)
            {
                T bt = Stack[index].PeekBottom(i).Item1;
                if (v.Equals(bt))
                    return true;
            }
            return false;
        }

        public void VisitNodes(Func<T, bool> func)
        {
            foreach (T v in graph.Vertices)
                Visited[v] = false;

            // Initialize all workers with
            // empty stack.
            Stack = new StackQueue<Tuple<T, StackQueue<T>>>[NumberOfWorkers];
            for (int i = 0; i < NumberOfWorkers; ++i)
                Stack[i] = new StackQueue<Tuple<T, StackQueue<T>>>(
                    new Tuple<T, StackQueue<T>>(default(T), new StackQueue<T>()));

            // Initialize first worker with stack containing all sources.
            foreach (T v in Source)
                Stack[0].PeekTop().Item2.Push(v);

            // Spawn workers.
            Parallel.For(0, NumberOfWorkers, (int index) =>
            {
                bool terminate = false;
                while (!terminate)
                {
                    T u = default(T);

                    GetWork(index);

                    while (Stack[index].Count >= 1 && Stack[index].PeekTop().Item2.Count > 0)
                    {
                        // There is stuff in the to do list. Pop it and perform dfs
                        // expansion of the vertex.
                        // Safe: No other threads will grab nodes within the cutoff,
                        // and no other threads can change this stack size.
                        StackQueue<T> todo = Stack[index].PeekTop().Item2;
                        u = todo.Pop();
                        Visited[u] = true;

                        // visit.
                        // yield return u;
                        //System.Console.WriteLine("visit " + u);
                        bool term = func(u);
                        if (term)
                        {
                            Terminate = true;
                            break;
                        }

                        // Push successors.
                        StackQueue<T> items = new StackQueue<T>();
                        foreach (T v in graph.ReverseSuccessors(u))
                        {
                            if (!Visited[v] && !SpecialContains(index, v))
                                items.Push(v);
                        }
                        if (items.Count != 0)
                        {
                            // Add new backtrack and to do list.
                            Stack[index].Push(
                                new Tuple<T, StackQueue<T>>(u, items));
                        }

                        // Synchronize threads on stack.
                        GetWork(index);
                    }

                    // Check for termination.
                    terminate = TerminateTest();
                }
            });
        }
    }
}
