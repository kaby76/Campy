namespace Campy.CIL
{
    using Campy.Graphs;
    using Mono.Cecil.Cil;
    using Mono.Cecil;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System;

    public class CIL_CFG : GraphLinkedList<int, CIL_CFG.Vertex, CIL_CFG.Edge>
    {
        private static int _node_number = 1;

        public int NewNodeNumber()
        {
            return _node_number++;
        }

        private List<CIL_CFG.Vertex> _entries = new List<Vertex>();

        public List<CIL_CFG.Vertex> Entries
        {
            get { return _entries; }
        }

        public CIL_CFG()
            : base()
        {
        }

        private Dictionary<object, List<CIL_CFG.Vertex>> _change_set = new Dictionary<object, List<Vertex>>();

        public void StartChangeSet(object observer)
        {
            if (_change_set.ContainsKey(observer))
            {
                _change_set[observer] = new List<Vertex>();
            }
            else
            {
                _change_set.Add(observer, new List<Vertex>());
            }
        }

        public List<CIL_CFG.Vertex> EndChangeSet(object observer)
        {
            if (_change_set.ContainsKey(observer))
            {
                List<CIL_CFG.Vertex> list = _change_set[observer];
                _change_set.Remove(observer);
                return list;
            }
            else
                return null;
        }

        public override GraphLinkedList<int, CIL_CFG.Vertex, CIL_CFG.Edge>.Vertex AddVertex(int v)
        {
            foreach (CIL_CFG.Vertex vertex in this.VertexNodes)
            {
                if (vertex.Name == v)
                    return vertex;
            }
            CIL_CFG.Vertex x = (Vertex)base.AddVertex(v);
            foreach (KeyValuePair<object, List<CIL_CFG.Vertex>> pair in this._change_set)
            {
                pair.Value.Add(x);
                Debug.Assert(_change_set[pair.Key].Contains(x));
            }
            return x;
        }

        public Vertex FindEntry(CIL_Inst cilInst)
        {
            Vertex result = null;

            // Find owning block.
            result = cilInst.Block;

            // Return entry block for method.
            return result.Entry;
        }

        public Vertex FindEntry(Mono.Cecil.Cil.Instruction inst)
        {
            Vertex result = null;
            foreach (CIL_CFG.Vertex node in this.VertexNodes)
                if (node.Instructions.First().Instruction == inst)
                    return node;
            return result;
        }

        public Vertex FindEntry(Mono.Cecil.MethodReference mr)
        {
            foreach (CIL_CFG.Vertex node in this.VertexNodes)
                if (node.Method == mr)
                    return node;
            return null;
        }

        public class Vertex
            : GraphLinkedList<int, Vertex, Edge>.Vertex
        {
            private MethodDefinition _method;

            public MethodDefinition Method
            {
                get
                {
                    return _method;
                }
                set
                {
                    _method = value;
                }
            }

            private List<CIL_Inst> _instructions = new List<CIL_Inst>();
            public List<CIL_Inst> Instructions
            {
                get
                {
                    return _instructions;
                }
            }
            protected int? _stack_level_in;
            protected int? _stack_level_out;
            protected int _stack_pre_last_instruction;

            public int? StackLevelIn
            {
                get
                {
                    return _stack_level_in;
                }
                set
                {
                    _stack_level_in = value;
                }
            }

            public int? StackLevelOut
            {
                get
                {
                    return _stack_level_out;
                }
                set
                {
                    _stack_level_out = value;
                }
            }

            public int StackLevelPreLastInstruction
            {
                get
                {
                    return _stack_pre_last_instruction;
                }
                set
                {
                    _stack_pre_last_instruction = value;
                }
            }


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
                    if (_entry == null)
                        return false;
                    if (_entry._ordered_list_of_blocks == null)
                        return false;
                    if (_entry._ordered_list_of_blocks.Count == 0)
                        return false;
                    if (_entry._ordered_list_of_blocks.First() != this)
                        return false;
                    return true;
                }
            }

            public bool IsCall
            {
                get
                {
                    CIL_Inst last = _instructions[_instructions.Count - 1];
                    switch (last.OpCode.FlowControl)
                    {
                        case FlowControl.Call:
                            return true;
                        default:
                            return false;
                    }
                }
            }

            public bool IsNewobj
            {
                get
                {
                    CIL_Inst last = _instructions[_instructions.Count - 1];
                    return last.OpCode.Code == Code.Newobj;
                }
            }

            public bool IsNewarr
            {
                get
                {
                    CIL_Inst last = _instructions[_instructions.Count - 1];
                    return last.OpCode.Code == Code.Newarr;
                }
            }

            public bool IsReturn
            {
                get
                {
                    CIL_Inst last = _instructions[_instructions.Count - 1];
                    switch (last.OpCode.FlowControl)
                    {
                        case FlowControl.Return:
                            return true;
                        default:
                            return false;
                    }
                }
            }

            private int _number_of_locals;
            private int _number_of_arguments;

            public int NumberOfLocals
            {
                get
                {
                    return _number_of_locals;
                }
                set
                {
                    _number_of_locals = value;
                }
            }

            public int NumberOfArguments
            {
                get
                {
                    return _number_of_arguments;
                }
                set
                {
                    _number_of_arguments = value;
                }
            }

            private bool _has_return;

            public bool HasReturnValue
            {
                get
                {
                    return _has_return;
                }
                set
                {
                    _has_return = value;
                }
            }

            public Vertex Exit
            {
                get
                {
                    List<Vertex> list = this._entry._ordered_list_of_blocks;
                    return list[list.Count() - 1];
                }
            }

            public List<Vertex> _ordered_list_of_blocks;

            public Vertex()
                : base()
            {
            }

            public void OutputEntireNode()
            {
                CIL_CFG.Vertex v = this;
                Console.WriteLine();
                Console.WriteLine("Node: " + v.Name + " ");
                Console.WriteLine(new String(' ', 4) + "Method " + v.Method.FullName);
                Console.WriteLine(new String(' ', 4) + "Args   " + v.NumberOfArguments);
                Console.WriteLine(new String(' ', 4) + "Locals " + v.NumberOfLocals);
                Console.WriteLine(new String(' ', 4) + "Return (reuse) " + v.HasReturnValue);
                if (this._Graph.Predecessors(v.Name).Any())
                {
                    Console.Write(new String(' ', 4) + "Edges from:");
                    foreach (object t in this._Graph.Predecessors(v.Name))
                    {
                        Console.Write(" " + t);
                    }
                    Console.WriteLine();
                }
                if (this._Graph.Successors(v.Name).Any())
                {
                    Console.Write(new String(' ', 4) + "Edges to:");
                    foreach (object t in this._Graph.Successors(v.Name))
                    {
                        Console.Write(" " + t);
                    }
                    Console.WriteLine();
                }
                Console.WriteLine(new String(' ', 4) + "Instructions:");
                foreach (CIL_Inst i in v._instructions)
                {
                    Console.Write(new String(' ', 8) + i + new String(' ', 4));
                    Console.WriteLine();
                }
                Console.WriteLine();
            }

            public Vertex Split(int i)
            {
                Debug.Assert(_instructions.Count != 0);
                // Split this node into two nodes, with all instructions after "i" in new node.
                CIL_CFG cfg = (CIL_CFG)this._Graph;
                Vertex result = (Vertex)cfg.AddVertex(cfg.NewNodeNumber());
                result.Method = this.Method;
                result.HasReturnValue = this.HasReturnValue;
                result._entry = this._entry;

                // Insert new block after this block.
                this._entry._ordered_list_of_blocks.Insert(
                    this._entry._ordered_list_of_blocks.IndexOf(this) + 1,
                    result);

                int count = _instructions.Count;

                // Add instructions from split point to new block.
                for (int j = i; j < count; ++j)
                {
                    CIL_Inst new_cil_inst = CIL_Inst.Wrap(_instructions[j].Instruction, result);
                    result._instructions.Add(new_cil_inst);
                }

                // Remove instructions from previous block.
                for (int j = i; j < count; ++j)
                {
                    this._instructions.RemoveAt(i);
                }

                Debug.Assert(this._instructions.Count != 0);
                Debug.Assert(result._instructions.Count != 0);
                Debug.Assert(this._instructions.Count + result._instructions.Count == count);

                CIL_Inst last_instruction = this._instructions[
                    this._instructions.Count - 1];

                // Transfer any out edges to pred block to new block.
                while (cfg.SuccessorNodes(this).Count() > 0)
                {
                    CIL_CFG.Vertex succ = cfg.SuccessorNodes(this).First();
                    cfg.DeleteEdge(this, succ);
                    cfg.AddEdge(result, succ);
                }

                // Add fall-through branch from pred to succ block.
                switch (last_instruction.OpCode.FlowControl)
                {
                    case FlowControl.Branch:
                        break;
                    case FlowControl.Break:
                        break;
                    case FlowControl.Call:
                        break;
                    case FlowControl.Cond_Branch:
                        cfg.AddEdge(this.Name, result.Name);
                        break;
                    case FlowControl.Meta:
                        break;
                    case FlowControl.Next:
                        cfg.AddEdge(this.Name, result.Name);
                        break;
                    case FlowControl.Phi:
                        break;
                    case FlowControl.Return:
                        break;
                    case FlowControl.Throw:
                        break;
                }

                //System.Console.WriteLine("After split");
                //cfg.Dump();
                //System.Console.WriteLine("-----------");
                return result;
            }
        }

        public class Edge
            : GraphLinkedList<int, CIL_CFG.Vertex, CIL_CFG.Edge>.Edge
        {
            public bool IsInterprocedural()
            {
                CIL_CFG.Vertex f = (CIL_CFG.Vertex)this.from;
                CIL_CFG.Vertex t = (CIL_CFG.Vertex)this.to;
                if (f.Method != t.Method)
                    return true;
                return false;
            }
        }

        public void OutputEntireGraph()
        {
            System.Console.WriteLine("Graph:");
            System.Console.WriteLine();
            System.Console.WriteLine("List of entries blocks:");
            System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Method");
            foreach (Vertex n in this._entries)
            {
                System.Console.Write("{0,8}", n);
                System.Console.Write(new string(' ', 4));
                System.Console.WriteLine(n.Method.FullName);
            }
            System.Console.WriteLine();
            System.Console.WriteLine("List of callers:");
            System.Console.WriteLine(new String(' ', 4) + "Node" + new string(' ', 4) + "Instruction");
            foreach (CIL_Inst caller in CIL_Inst.CallInstructions)
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

            foreach (Vertex n in VertexNodes)
            {
                if (n._ordered_list_of_blocks != null)
                {
                    foreach (Vertex v in n._ordered_list_of_blocks)
                    {
                        v.OutputEntireNode();
                    }
                }
            }
        }

        class CallEnumerator : IEnumerable<CIL_CFG.Vertex>
        {
            CIL_CFG.Vertex _node;

            public CallEnumerator(CIL_CFG.Vertex node)
            {
                _node = node;
            }

            public IEnumerator<CIL_CFG.Vertex> GetEnumerator()
            {
                foreach (CIL_CFG.Vertex current in _node._ordered_list_of_blocks)
                {
                    foreach (CIL_CFG.Vertex next in _node._Graph.SuccessorNodes(current))
                    {
                        if (next.IsEntry && next.Method != _node.Method)
                            yield return next;
                    }
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public IEnumerable<CIL_CFG.Vertex> AllInterproceduralCalls(CIL_CFG.Vertex node)
        {
            return new CallEnumerator(node);
        }

    }
}
