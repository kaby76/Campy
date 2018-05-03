using Mono.Collections.Generic;
using Swigged.LLVM;

namespace Campy.Compiler
{
    using Campy.Graphs;
    using Mono.Cecil.Cil;
    using Mono.Cecil;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System;

    public class CFG : GraphAdjList<CFG.Vertex, CFG.Edge>
    {
        private static int _node_number = 1;

        public int NewNodeNumber()
        {
            return _node_number++;
        }

        private List<CFG.Vertex> _entries = new List<Vertex>();

        public List<CFG.Vertex> Entries
        {
            get { return _entries; }
        }

        public CFG()
            : base()
        {
        }

        private Dictionary<int, List<CFG.Vertex>> _change_set = new Dictionary<int, List<Vertex>>();
        private Random random;
        
        public int StartChangeSet()
        {
            if (random == null)
            {
                random = new Random();
            }
            int new_num = 0;
            for (;;)
            {
                new_num = random.Next(100000000);
                if (_change_set.ContainsKey(new_num))
                {
                    continue;
                }
                break;
            }
            _change_set.Add(new_num, new List<Vertex>());
            return new_num;
        }

        public List<CFG.Vertex> PopChangeSet(int num)
        {
            if (_change_set.ContainsKey(num))
            {
                List<CFG.Vertex> list = _change_set[num];
                _change_set.Remove(num);
                return list;
            }
            throw new Exception("Unknown change set.");
        }

        public override CFG.Vertex AddVertex(CFG.Vertex v)
        {
            foreach (CFG.Vertex vertex in this.Vertices)
            {
                if (vertex == v)
                    return vertex;
            }
            CFG.Vertex x = (Vertex)base.AddVertex(v);
            x._graph = this;
            foreach (KeyValuePair<int, List<CFG.Vertex>> pair in this._change_set)
            {
                pair.Value.Add(x);
                Debug.Assert(_change_set[pair.Key].Contains(x));
            }
            return x;
        }

        public Vertex FindEntry(INST inst)
        {
            Vertex result = null;

            // Find owning block.
            result = inst.Block;

            // Return entry block for method.
            return result.Entry;
        }

        public Vertex FindEntry(Mono.Cecil.Cil.Instruction inst)
        {
            Vertex result = null;
            foreach (CFG.Vertex node in this.Vertices)
                if (node.Instructions.First().Instruction == inst)
                    return node;
            return result;
        }

        public Vertex FindEntry(Mono.Cecil.MethodReference mr)
        {
            foreach (CFG.Vertex node in this.Vertices)
                if (node._original_method_reference == mr)
                    return node;
            return null;
        }

        public class Vertex
        {

            public struct LLVMINFO
            {
                public BasicBlockRef BasicBlock { get; set; }
                public ValueRef MethodValueRef { get; set; }
                public BuilderRef Builder { get; set; }
                public ModuleRef Module { get; set; }
                private List<ValueRef> ph;
                public List<ValueRef> Phi
                {
                    get
                    {
                        if (ph == null)
                            ph = new List<ValueRef>();
                        return ph;
                    }
                }
            }

            public string Name { get; set; }
            public override string ToString()
            {
                return Name;
            }
            public CFG.Vertex OriginalVertex { get; set; }
            public Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> OpsFromOriginal { get; set; } = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();
            public CFG.Vertex PreviousVertex { get; set; }
            public Tuple<Tuple<TypeReference, GenericParameter>, System.Type> OpFromPreviousNode { get; set; }
            public MethodReference RewrittenCalleeSignature { get; set; }
            public List<INST> Instructions { get; set; } = new List<INST>();
            public CFG _graph
            {
                get;
                set;
            }
            public LLVMINFO LlvmInfo;
            public bool AlreadyCompiled { get; set; }
            public MethodDefinition _method_definition { get; set; }
            public MethodReference _original_method_reference { get; set; }
            public bool HasThis
            {
                get;
                set;
            }
            public bool HasScalarReturnValue { get; set; }
            public bool HasStructReturnValue { get; set; }
            //public STATE StateIn { get; set; }
            //public STATE StateOut { get; set; }
            private Vertex _entry;
            public Vertex Entry
            {
                get { return _entry; }
                set { _entry = value; }
            }
            public bool IsEntry
            {
                get
                {
                    return this.Entry == this;
                }
            }
            public int StackNumberOfLocals
            {
                get;
                set;
            }
            public int StackNumberOfArguments
            {
                get;
                set;
            }
            public Vertex Exit
            {
                get
                {
                    return null;
                }
            }

            public Vertex()
                : base()
            {
            }

            public Vertex(Vertex copy)
                : base()
            {
                _original_method_reference = copy._original_method_reference;
                this.Instructions = copy.Instructions;
                HasScalarReturnValue = copy.HasScalarReturnValue;

            }

            public void OutputEntireNode()
            {
                if (!Campy.Utils.Options.IsOn("graph_trace"))
                    return;

                CFG.Vertex v = this;
                Console.WriteLine();
                Console.WriteLine("Node: " + v.ToString() + " ");
                Console.WriteLine(new String(' ', 4) + "Method " + v._original_method_reference.FullName + " " + v._original_method_reference.Module.Name + " " + v._original_method_reference.Module.FileName);
                Console.WriteLine(new String(' ', 4) + "Method " + v._method_definition.FullName + " " + v._method_definition.Module.Name + " " + v._method_definition.Module.FileName);
                Console.WriteLine(new String(' ', 4) + "HasThis   " + v.HasThis);
                Console.WriteLine(new String(' ', 4) + "Args   " + v.StackNumberOfArguments);
                Console.WriteLine(new String(' ', 4) + "Locals " + v.StackNumberOfLocals);
                Console.WriteLine(new String(' ', 4) + "Return (reuse) " + v.HasScalarReturnValue);
                if (this._graph.Predecessors(v).Any())
                {
                    Console.Write(new String(' ', 4) + "Edges from:");
                    foreach (object t in this._graph.Predecessors(v))
                    {
                        Console.Write(" " + t);
                    }
                    Console.WriteLine();
                }
                if (this._graph.Successors(v).Any())
                {
                    Console.Write(new String(' ', 4) + "Edges to:");
                    foreach (object t in this._graph.Successors(v))
                    {
                        Console.Write(" " + t);
                    }
                    Console.WriteLine();
                }
                Console.WriteLine(new String(' ', 4) + "Instructions:");
                foreach (INST i in v.Instructions)
                {
                    Console.Write(new String(' ', 8) + i + new String(' ', 4));
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }

        public class Edge : DirectedEdge<CFG.Vertex>
        {
            public Edge()
                : base(null, null)
            {
            }

            public bool IsInterprocedural()
            {
                CFG.Vertex f = (CFG.Vertex)this.From;
                CFG.Vertex t = (CFG.Vertex)this.To;
                if (f._original_method_reference != t._original_method_reference)
                    return true;
                return false;
            }
        }

        public void OutputEntireGraph()
        {
            if (!Campy.Utils.Options.IsOn("graph_trace"))
                return;
            System.Console.WriteLine("Graph:");
            System.Console.WriteLine();
            System.Console.WriteLine("List of entry blocks:");
            System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Method");
            foreach (Vertex n in this._entries)
            {
                System.Console.Write("{0,8}", n);
                System.Console.Write(new string(' ', 4));
                System.Console.WriteLine(n._original_method_reference.FullName);
            }
            System.Console.WriteLine();
            System.Console.WriteLine("List of callers:");
            System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Instruction");
            foreach (INST caller in INST.CallInstructions)
            {
                Vertex n = caller.Block;
                System.Console.Write("{0,8}", n);
                System.Console.Write(new string(' ', 4));
                System.Console.WriteLine(caller);
            }
            if (this._entries.Any())
            {
                System.Console.WriteLine();
                System.Console.WriteLine("List of orphan blocks:");
                System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Method");
                System.Console.WriteLine();
            }

            foreach (Vertex n in Vertices)
            {
                  n.OutputEntireNode();
            }
        }

        public void OutputDotGraph()
        {
            if (!Campy.Utils.Options.IsOn("dot_graph"))
                return;

            Dictionary<CFG.Vertex,bool> visited = new Dictionary<CFG.Vertex, bool>();
            System.Console.WriteLine("digraph {");
            foreach (var n in this.Edges)
            {
                System.Console.WriteLine(n.From + " -> " + n.To + ";");
                visited[n.From] = true;
                visited[n.To] = true;
            }
            foreach (var n in this.Vertices)
            {
                if (visited.ContainsKey(n)) continue;
                System.Console.WriteLine(n + ";");
            }
            System.Console.WriteLine("}");
            System.Console.WriteLine();
        }

        class CallEnumerator : IEnumerable<CFG.Vertex>
        {
            CFG.Vertex _node;

            public CallEnumerator(CFG.Vertex node)
            {
                _node = node;
            }

            public IEnumerator<CFG.Vertex> GetEnumerator()
            {
                // Create a sorted list of nodes for given node.
                List<CFG.Vertex> list = new List<Vertex>();
                foreach (CFG.Vertex v in _node._graph.Vertices)
                {
                    if (v.Entry == _node) list.Add(v);
                }
                for (int i = 0; i < list.Count - 1; ++i)
                {
                    var v1 = list[i];
                    var v2 = list[i + 1];
                    var sv1 = v1.Instructions.First().ToString();
                    var sv2 = v2.Instructions.First().ToString();
                    if (sv1.CompareTo(sv2) > 0)
                    {
                        list[i] = v2;
                        list[i + 1] = v1;
                    }
                }
                foreach (CFG.Vertex current in list)
                {
                    foreach (CFG.Vertex next in _node._graph.SuccessorNodes(current))
                    {
                        if (next.IsEntry && next._original_method_reference != _node._original_method_reference)
                            yield return next;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<CFG.Vertex> AllInterproceduralCalls(CFG.Vertex node)
        {
            return new CallEnumerator(node);
        }

    }
}
