namespace Campy.ControlFlowGraph
{
    using Campy.Utils;
    using Mono.Cecil;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq.Expressions;
    using System.Linq;
    using System.Reflection;
    using System;
    using Campy.Types;

    public class Reader
    {
        private CFG _cfg = new CFG();

        public CFG Cfg
        {
            get { return _cfg; }
            set { _cfg = value; }
        }
        public void AnalyzeThisAssembly()
        {
            if (Environment.Is64BitProcess)
            {
            }
            System.Diagnostics.StackTrace stack_trace = new System.Diagnostics.StackTrace(true);
            System.Diagnostics.StackFrame stack_frame = stack_trace.GetFrame(1);
            System.Reflection.Assembly assembly = stack_frame.GetMethod().DeclaringType.Assembly;
            this.AddAssembly(assembly);
            this.ExtractBasicBlocks();
            _cfg.OutputEntireGraph();
        }

        public void AnalyzeMethod(Campy.Types._Kernel_type expr)
        {
            MethodInfo methodInfo = expr.Method;
            //var methodInfo = ((MethodCallExpression)expr.Body).Method;
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
            _cfg.OutputEntireGraph();
            _cfg.OutputDotGraph();
        }

        public void AnalyzeMethod(MethodInfo methodInfo)
        {
            //var methodInfo = ((MethodCallExpression)expr.Body).Method;
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
        }

        public void AnalyzeMethod(Mono.Cecil.MethodDefinition md)
        {
            this.Add(md);
            this.ExtractBasicBlocks();
        }


        public void Add(System.Type type)
        {
            // Add all methods of type.
            BindingFlags findFlags = BindingFlags.NonPublic |
                                                BindingFlags.Public |
                                                BindingFlags.Static |
                                                BindingFlags.Instance |
                                                BindingFlags.InvokeMethod |
                                                BindingFlags.OptionalParamBinding |
                                                BindingFlags.DeclaredOnly;
            foreach (System.Reflection.MethodInfo definition in type.GetMethods(findFlags))
                Add(definition);
        }

        public void Add(MethodInfo reference)
        {
            Mono.Cecil.MethodDefinition definition = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilMethodDefinition(reference);
            Add(definition);
        }

        public void AddAssembly(String assembly_file_name)
        {
            Mono.Cecil.ModuleDefinition module = LoadAssembly(assembly_file_name);
            String full_name = System.IO.Path.GetFullPath(assembly_file_name);
            foreach (Mono.Cecil.ModuleDefinition md in this._analyzed_modules)
                if (md.FullyQualifiedName.Equals(full_name))
                    return;
            _analyzed_modules.Add(module);
            StackQueue<Mono.Cecil.TypeDefinition> type_definitions = new StackQueue<Mono.Cecil.TypeDefinition>();
            StackQueue<Mono.Cecil.TypeDefinition> type_definitions_closure = new StackQueue<Mono.Cecil.TypeDefinition>();
            foreach (Mono.Cecil.TypeDefinition td in module.Types)
            {
                type_definitions.Push(td);
            }
            while (type_definitions.Count > 0)
            {
                Mono.Cecil.TypeDefinition ty = type_definitions.Pop();
                type_definitions_closure.Push(ty);
                foreach (Mono.Cecil.TypeDefinition ntd in ty.NestedTypes)
                    type_definitions.Push(ntd);
            }
            foreach (Mono.Cecil.TypeDefinition td in type_definitions_closure)
                foreach (Mono.Cecil.MethodDefinition definition in td.Methods)
                    Add(definition);
        }

        public void Add(Mono.Cecil.TypeReference type)
        {
            // Add all methods of type.
            Mono.Cecil.TypeDefinition type_defintion = type.Resolve();
            foreach (Mono.Cecil.MethodDefinition definition in type_defintion.Methods)
                Add(definition);
        }

        public void Add(Mono.Cecil.MethodReference reference)
        {
            Add(reference.Resolve());
        }

        public void Add(Mono.Cecil.MethodDefinition definition)
        {
            if (_methoddefs_done.Contains(definition))
                return;
            if (_methoddefs_to_do.Contains(definition))
                return;
            _methoddefs_to_do.Push(definition);
        }

        public void AddAssembly(Assembly assembly)
        {
            String assembly_file_name = assembly.Location;
            AddAssembly(assembly_file_name);
        }

        public void ExtractBasicBlocks()
        {
            while (_methoddefs_to_do.Count > 0)
            {
                int change_set_id = this.Cfg.StartChangeSet();

                Mono.Cecil.MethodDefinition definition = _methoddefs_to_do.Pop();
                ExtractBasicBlocksOfMethod(definition);

                var blocks = this.Cfg.PopChangeSet(change_set_id);

                // Get closure of calls, if possible.
                foreach (var b in blocks)
                {
                    foreach (Inst i in b.Instructions)
                    {
                        var fc = i.OpCode.FlowControl;
                        if (fc != Mono.Cecil.Cil.FlowControl.Call)
                            continue;
                        object method = i.Operand;
                        if (method as Mono.Cecil.MethodReference != null)
                        {
                            Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
                            Mono.Cecil.MethodDefinition md = mr.Resolve();
                            Add(md);
                        }
                    }
                }
            }
        }

        private List<Mono.Cecil.ModuleDefinition> _loaded_modules = new List<ModuleDefinition>();
        private List<Mono.Cecil.ModuleDefinition> _analyzed_modules = new List<ModuleDefinition>();
        // Everything revolves around the defined methods in Mono. The "to do" list contains
        // all methods to convert into a CFG and then to SSA.
        private StackQueue<Mono.Cecil.MethodDefinition> _methoddefs_to_do = new StackQueue<Mono.Cecil.MethodDefinition>();
        private List<Mono.Cecil.MethodDefinition> _methoddefs_done = new List<MethodDefinition>();
        private static MethodInfo GetMethodInfo(Action a)
        {
            return a.Method;
        }

        private Mono.Cecil.ModuleDefinition LoadAssembly(String assembly_file_name)
        {
            String full_name = System.IO.Path.GetFullPath(assembly_file_name);
            foreach (Mono.Cecil.ModuleDefinition md in this._loaded_modules)
                if (md.FullyQualifiedName.Equals(full_name))
                    return md;
            Mono.Cecil.ModuleDefinition module = ModuleDefinition.ReadModule(assembly_file_name);
            _loaded_modules.Add(module);
            return module;
        }

        private void ExtractBasicBlocksOfMethod(MethodDefinition definition)
        {
            _methoddefs_done.Add(definition);

            // Make sure definition assembly is loaded. The analysis of the method cannot
            // be done if the routine hasn't been loaded into Mono!
            String full_name = definition.Module.FullyQualifiedName;
            LoadAssembly(full_name);
            if (definition.Body == null)
            {
                System.Console.WriteLine("WARNING: METHOD BODY NULL! " + definition);
                return;
            }
            int instruction_count = definition.Body.Instructions.Count;
            StackQueue<Mono.Cecil.Cil.Instruction> leader_list = new StackQueue<Mono.Cecil.Cil.Instruction>();

            // Each method is a leader of a block.
            CFG.Vertex v = (CFG.Vertex)_cfg.AddVertex(_cfg.NewNodeNumber());
            v.Method = definition;
            v.HasReturnValue = definition.IsReuseSlot;
            v.Entry = v;
            _cfg.Entries.Add(v);
            for (int j = 0; j < instruction_count; ++j)
            {
                // accumulate jump to locations since these split blocks.
                Mono.Cecil.Cil.Instruction mi = definition.Body.Instructions[j];
                //System.Console.WriteLine(mi);
                Inst i = Inst.Wrap(mi);
                i.Block = v;
                Mono.Cecil.Cil.OpCode op = i.OpCode;
                Mono.Cecil.Cil.FlowControl fc = op.FlowControl;

                v.Instructions.Add(i);

                if (fc == Mono.Cecil.Cil.FlowControl.Next)
                    continue;
                if (fc == Mono.Cecil.Cil.FlowControl.Branch
                    || fc == Mono.Cecil.Cil.FlowControl.Cond_Branch)
                {
                    // Save leader target of branch.
                    object o = i.Operand;
                    // Two cases that I know of: operand is just and instruction,
                    // or operand is an array of instructions (via a switch instruction).

                    Mono.Cecil.Cil.Instruction oo = o as Mono.Cecil.Cil.Instruction;
                    Mono.Cecil.Cil.Instruction[] ooa = o as Mono.Cecil.Cil.Instruction[];
                    if (oo != null)
                    {
                        leader_list.Push(oo);
                    }
                    else if (ooa != null)
                    {
                        foreach (Mono.Cecil.Cil.Instruction ins in ooa)
                        {
                            Debug.Assert(ins != null);
                            leader_list.Push(ins);
                        }
                    }
                    else
                        throw new Exception("Unknown operand type for basic block partitioning.");

                }
            }
            StackQueue<int> ordered_leader_list = new StackQueue<int>();
            for (int j = 0; j < instruction_count; ++j)
            {
                // Order jump targets. These denote locations
                // where to split blocks. However, it's ordered,
                // so that splitting is done from last instruction in block
                // to first instruction in block.
                Mono.Cecil.Cil.Instruction i = definition.Body.Instructions[j];
                //System.Console.WriteLine("Looking for " + i);
                if (leader_list.Contains(i))
                    ordered_leader_list.Push(j);
            }

            // Split block at jump targets in reverse.
            while (ordered_leader_list.Count > 0)
            {
                int i = ordered_leader_list.Pop();
                CFG.Vertex new_node = v.Split(i);
            }

            //this.Dump();

            StackQueue<CFG.Vertex> stack = new StackQueue<CFG.Vertex>();
            foreach (CFG.Vertex node in _cfg.VertexNodes) stack.Push(node);
            while (stack.Count > 0)
            {
                // Split blocks at branches, not including calls, with following
                // instruction a leader of new block.
                CFG.Vertex node = stack.Pop();
                int node_instruction_count = node.Instructions.Count;
                for (int j = 0; j < node_instruction_count; ++j)
                {
                    Inst i = node.Instructions[j];
                    Mono.Cecil.Cil.OpCode op = i.OpCode;
                    Mono.Cecil.Cil.FlowControl fc = op.FlowControl;
                    if (fc == Mono.Cecil.Cil.FlowControl.Next)
                        continue;
                    if (fc == Mono.Cecil.Cil.FlowControl.Call)
                        continue;
                    if (fc == Mono.Cecil.Cil.FlowControl.Meta)
                        continue;
                    if (fc == Mono.Cecil.Cil.FlowControl.Phi)
                        continue;
                    if (j + 1 >= node_instruction_count)
                        continue;
                    CFG.Vertex new_node = node.Split(j + 1);
                    stack.Push(new_node);
                    break;
                }
            }

            //this.Dump();
            stack = new StackQueue<CFG.Vertex>();
            foreach (CFG.Vertex node in _cfg.VertexNodes) stack.Push(node);
            while (stack.Count > 0)
            {
                // Add in all final non-fallthrough branch edges.
                CFG.Vertex node = stack.Pop();
                int node_instruction_count = node.Instructions.Count;
                Inst i = node.Instructions[node_instruction_count - 1];
                Mono.Cecil.Cil.OpCode op = i.OpCode;
                Mono.Cecil.Cil.FlowControl fc = op.FlowControl;
                switch (fc)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                        {
                            // Two cases: i.Operand is a single instruction, or an array of instructions.
                            if (i.Operand as Mono.Cecil.Cil.Instruction != null)
                            {
                                Mono.Cecil.Cil.Instruction target_instruction = i.Operand as Mono.Cecil.Cil.Instruction;
                                CFG.Vertex target_node = _cfg.VertexNodes.First(
                                    (CFG.Vertex x) =>
                                    {
                                        if (!x.Instructions.First().Instruction.Equals(target_instruction))
                                            return false;
                                        return true;
                                    });
                                _cfg.AddEdge(node, target_node);
                            }
                            else if (i.Operand as Mono.Cecil.Cil.Instruction[] != null)
                            {
                                foreach (Mono.Cecil.Cil.Instruction target_instruction in (i.Operand as Mono.Cecil.Cil.Instruction[]))
                                {
                                    CFG.Vertex target_node = _cfg.VertexNodes.First(
                                        (CFG.Vertex x) =>
                                        {
                                            if (!x.Instructions.First().Instruction.Equals(target_instruction))
                                                return false;
                                            return true;
                                        });
                                    System.Console.WriteLine("Create edge a " + node.Name + " to " + target_node.Name);
                                    _cfg.AddEdge(node, target_node);
                                }
                            }
                            else
                                throw new Exception("Unknown operand type for conditional branch.");
                            break;
                        }
                    case Mono.Cecil.Cil.FlowControl.Break:
                        break;
                    case Mono.Cecil.Cil.FlowControl.Call:
                        {
                            // We no longer split at calls. Splitting causes
                            // problems because interprocedural edges are
                            // produced. That's not good because it makes
                            // code too "messy".
                            break;

                            //object o = i.Operand;
                            //if (o as Mono.Cecil.MethodReference != null)
                            //{
                            //    Mono.Cecil.MethodReference r = o as Mono.Cecil.MethodReference;
                            //    Mono.Cecil.MethodDefinition d = r.Resolve();
                            //    IEnumerable<CFG.Vertex> target_node_list = _cfg.VertexNodes.Where(
                            //        (CFG.Vertex x) =>
                            //        {
                            //            return x.Method.FullName == r.FullName
                            //                && x.Entry == x;
                            //        });
                            //    int c = target_node_list.Count();
                            //    if (c >= 1)
                            //    {
                            //        // target_node is the entry for a method. Also get the exit.
                            //        CFG.Vertex target_node = target_node_list.First();
                            //        CFG.Vertex exit_node = target_node.Exit;
                            //    }
                            //}
                            //break;
                        }
                    case Mono.Cecil.Cil.FlowControl.Meta:
                        break;
                    case Mono.Cecil.Cil.FlowControl.Next:
                        break;
                    case Mono.Cecil.Cil.FlowControl.Phi:
                        break;
                    case Mono.Cecil.Cil.FlowControl.Return:
                        break;
                    case Mono.Cecil.Cil.FlowControl.Throw:
                        break;
                }
            }

            //this.Dump();

            // At this point, the given method has been converted into a bunch of basic blocks,
            // and are part of the CFG.
        }
    }
}
