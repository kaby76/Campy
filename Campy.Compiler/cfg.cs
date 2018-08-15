using Mono.Cecil.Cil;

namespace Campy.Compiler
{
    using Campy.Graphs;
    using Mono.Cecil;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System;
    using Swigged.LLVM;

    public class CFG : GraphAdjList<CFG.Vertex, CFG.Edge>
    {
        private static int _node_number = 1;
        private Dictionary<int, List<Vertex>> _change_set = new Dictionary<int, List<Vertex>>();
        private Random random;

        public List<Vertex> Entries { get; } = new List<Vertex>();

        public int NewNodeNumber()
        {
            return _node_number++;
        }

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

        public List<Vertex> PopChangeSet(int num)
        {
            if (_change_set.ContainsKey(num))
            {
                List<CFG.Vertex> list = _change_set[num];
                _change_set.Remove(num);
                return list;
            }
            throw new Exception("Unknown change set.");
        }

        public List<Vertex> PeekChangeSet(int num)
        {
            if (_change_set.ContainsKey(num))
            {
                List<CFG.Vertex> list = _change_set[num];
                return list;
            }
            throw new Exception("Unknown change set.");
        }

        public override Vertex AddVertex(Vertex v)
        {
            foreach (var vertex in Vertices)
            {
                if (vertex == v) return vertex;
            }
            var x = base.AddVertex(v);
            x._graph = this;
            foreach (KeyValuePair<int, List<Vertex>> pair in this._change_set)
            {
                pair.Value.Add(x);
                Debug.Assert(_change_set[pair.Key].Contains(x));
            }
            return x;
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
            public Dictionary<int, bool> local_alloc = new Dictionary<int, bool>();
            public Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> OpsFromOriginal { get; set; } = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();
            public Tuple<Tuple<TypeReference, GenericParameter>, System.Type> OpFromPreviousNode { get; set; }
            public List<INST> Instructions { get; set; } = new List<INST>();
            public CFG _graph { get; set; }
            public LLVMINFO LlvmInfo;
            public bool AlreadyCompiled { get; set; }
            // Method reference, which might be a generic instance.
            public MethodReference _method_reference { get; set; }
            // Cached method definition so I don't need to do _method_referenc.Resolve() all the time.
            public MethodDefinition _method_definition { get; set; }
            // This may be the original reference in the code before substitution with BCL code.
            public MethodReference _original_method_reference { get; set; }
            public bool HasThis { get; set; }
            public bool HasScalarReturnValue { get; set; }
            public bool HasStructReturnValue { get; set; }
            public Vertex Entry { get; set; }
            public bool IsEntry { get { return Entry == this; } }
            public bool IsCatch { get; set; }
            public Mono.Cecil.Cil.ExceptionHandler ExceptionHandler { get; set; }
            public TypeReference CatchType { get; set; }
            public int StackNumberOfLocals { get; set; }
            public int StackNumberOfArguments { get; set; }

            public void OutputEntireNode()
            {
                if (!Campy.Utils.Options.IsOn("graph_trace"))
                    return;

                var v = this;
                Console.WriteLine();
                Console.WriteLine("Node: " + v.ToString() + " ");
                Console.WriteLine(new String(' ', 4) + "Method " + v._original_method_reference.FullName + " " + v._original_method_reference.Module.Name + " " + v._original_method_reference.Module.FileName);
                Console.WriteLine(new String(' ', 4) + "Method " + v._method_definition.FullName + " " + v._method_definition.Module.Name + " " + v._method_definition.Module.FileName);
                Console.WriteLine(new String(' ', 4) + "HasThis   " + v.HasThis);
                Console.WriteLine(new String(' ', 4) + "Args   " + v.StackNumberOfArguments);
                Console.WriteLine(new String(' ', 4) + "Locals " + v.StackNumberOfLocals);
                Console.WriteLine(new String(' ', 4) + "Return (reuse) " + v.HasScalarReturnValue);
                if (_graph.Predecessors(v).Any())
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

        public class Edge : DirectedEdge<Vertex>
        {
            public Edge()
                : base(null, null)
            {
            }
        }

        public void OutputEntireGraph()
        {
            System.Console.WriteLine("Graph:");
            System.Console.WriteLine();
            System.Console.WriteLine("List of entry blocks:");
            System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Method");
            foreach (var n in Entries)
            {
                System.Console.Write("{0,8}", n);
                System.Console.Write(new string(' ', 4));
                System.Console.WriteLine(n._original_method_reference.FullName);
            }
            System.Console.WriteLine();
            System.Console.WriteLine("List of callers:");
            System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Instruction");
            foreach (var caller in INST.CallInstructions)
            {
                Vertex n = caller.Block;
                System.Console.Write("{0,8}", n);
                System.Console.Write(new string(' ', 4));
                System.Console.WriteLine(caller);
            }
            if (Entries.Any())
            {
                System.Console.WriteLine();
                System.Console.WriteLine("List of orphan blocks:");
                System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Method");
                System.Console.WriteLine();
            }

            foreach (var n in Vertices)
            {
                  n.OutputEntireNode();
            }
        }

        public void OutputDotGraph()
        {
            Dictionary<CFG.Vertex,bool> visited = new Dictionary<CFG.Vertex, bool>();
            Console.WriteLine("digraph {");
            foreach (var n in Edges)
            {
                Console.WriteLine(n.From + " -> " + n.To + ";");
                visited[n.From] = true;
                visited[n.To] = true;
            }
            foreach (var n in Vertices)
            {
                if (visited.ContainsKey(n)) continue;
                Console.WriteLine(n + ";");
            }
            Console.WriteLine("}");
            Console.WriteLine();
        }
    }
}
