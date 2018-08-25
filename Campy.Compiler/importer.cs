
namespace Campy.Compiler
{
    using Mono.Cecil.Cil;
    using Mono.Cecil.Rocks;
    using Mono.Cecil;
    using Mono.Collections.Generic;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System;
    using Utils;
    using Campy.Meta;

    internal class IMPORTER
    {
        private StackQueue<MethodReference> _methods_to_do;
        private List<string> _methods_done;

        public bool Failed
        {
            get; internal set;
        }

        public CFG Cfg
        {
            get; internal set;
        }

        private static IMPORTER _instance;

        public static IMPORTER Singleton()
        {
            if (_instance == null)
            {
                _instance = new IMPORTER();
            }
            return _instance;
        }

        private IMPORTER()
        {
            Cfg = new CFG();
            _methods_to_do = new StackQueue<MethodReference>();
            _methods_done = new List<string>();
            _methods_avoid.Add("System.Void System.ThrowHelper::ThrowArgumentOutOfRangeException()");
            _methods_avoid.Add("System.Void System.ThrowHelper::ThrowArgumentOutOfRangeException()");
            _methods_avoid.Add("System.Void System.ArgumentOutOfRangeException::.ctor(System.String, System.String)");
        }

        public void Add(ModuleDefinition module)
        {
            StackQueue<TypeDefinition> type_definitions = new StackQueue<TypeDefinition>();
            StackQueue<TypeDefinition> type_definitions_closure = new StackQueue<TypeDefinition>();
            foreach (TypeDefinition td in module.Types)
                type_definitions.Push(td);
            while (type_definitions.Count > 0)
            {
                TypeDefinition ty = type_definitions.Pop();
                type_definitions_closure.Push(ty);
                foreach (TypeDefinition ntd in ty.NestedTypes)
                    type_definitions.Push(ntd);
            }
            foreach (TypeDefinition td in type_definitions_closure)
                foreach (MethodDefinition definition in td.Methods)
                    Add(definition);
        }

        public void AnalyzeMethod(MethodReference method_reference)
        {
            Failed = false; // Might be dubious to reset here.
            this.Add(method_reference);
            this.ExtractBasicBlocks();
            Cfg.OutputEntireGraph();
            Cfg.OutputDotGraph();
        }

        internal void Add(MethodReference method_reference)
        {
            // Do not analyze if nonsense.
            if (method_reference == null) return;
            // Do not analyze if already being or has been considered.
            if (_methods_avoid.Contains(method_reference.FullName)) return;
            if (_methods_done.Contains(method_reference.FullName)) return;
            foreach (var tuple in _methods_to_do)
            {
                if (tuple.FullName == method_reference.FullName) return;
            }

            if (method_reference.ContainsGenericParameter)
            {
                throw new Exception("method " + method_reference.FullName + " contains generic parameter.");
            }

            foreach (var p in method_reference.Parameters)
            {
                if (p.ParameterType.ContainsGenericParameter)
                    throw new Exception("method " + method_reference.FullName + " contains generic parameter.");
            }

            if (Campy.Utils.Options.IsOn("overview_import_computation_trace"))
            {
                System.Console.WriteLine(" -> " + method_reference.FullName);
            }

            _methods_to_do.Push(method_reference);
        }

        public void Add(MethodInfo reference)
        {
            String kernel_assembly_file_name = reference.DeclaringType.Assembly.Location;
            Mono.Cecil.ModuleDefinition md = Campy.Meta.StickyReadMod.StickyReadModule(kernel_assembly_file_name);
            MethodReference refer = md.ImportReference(reference);
            Add(refer);
        }

        public void Add(Type type)
        {
            var mono_type = type.ToMonoTypeReference();
            foreach (var m in mono_type.Resolve().Methods)
            {
                Add(m);
            }
        }

        private List<string> _methods_avoid = new List<string>();

        private void ExtractBasicBlocks()
        {
            while (_methods_to_do.Count > 0)
            {
                int change_set_id = this.Cfg.StartChangeSet();
                MethodReference reference = _methods_to_do.Pop();
                if (Campy.Utils.Options.IsOn("overview_import_computation_trace"))
                    System.Console.WriteLine("Importing " + reference.FullName);
                ExtractBasicBlocksOfMethod(reference);
                var blocks = this.Cfg.PopChangeSet(change_set_id);
                blocks.ComputeBasicMethodProperties();
                blocks.PropagateTypesAndPerformCallClosure();
            }
        }

        private static ModuleDefinition LoadAssembly(String assembly_file_name)
        {
            // Microsoft keeps screwing and changing the code that finds assemblies all the damn time.
            // Get the type of Importer. Assume that it's in the same directory as everything else.
            // Set that path for the god damn resolver.
            var fucking_directory_path = Campy.Utils.CampyInfo.PathOfCampy();
            var full_frigging_path_of_assembly_file = fucking_directory_path + "\\" + assembly_file_name;
            ModuleDefinition module = Campy.Meta.StickyReadMod.StickyReadModule(
                assembly_file_name,
                new ReaderParameters { ReadSymbols = true});
            return module;
        }

        private void ExtractBasicBlocksOfMethod(MethodReference method_reference)
        {
            MethodReference original_method_reference = method_reference;
            _methods_done.Add(original_method_reference.FullName);

  //          MethodReference new_method_reference = original_method_reference.SubstituteMethod2();
  //          method_reference = new_method_reference != null ? new_method_reference : original_method_reference;

            MethodDefinition method_definition = method_reference.Resolve();
            if (method_definition == null || method_definition.Body == null)
                return;
            int change_set = Cfg.StartChangeSet();
            int instruction_count = method_definition.Body.Instructions.Count;
            List<INST> split_point = new List<INST>();
            CFG.Vertex basic_block = (CFG.Vertex)Cfg.AddVertex(new CFG.Vertex(){Name = Cfg.NewNodeNumber().ToString()});
            basic_block._method_definition = method_definition;
            basic_block._method_reference = original_method_reference;
            basic_block.Entry = basic_block;
            Cfg.Entries.Add(basic_block);

            // Add instructions to the basic block, including debugging information.
            // First, get debugging information on line/column/offset in method.
            if (!original_method_reference.Module.HasSymbols)
            {
                // Try to get symbols, but if none available, don't worry about it.
                try { original_method_reference.Module.ReadSymbols(); } catch { }
            }
            var symbol_reader = original_method_reference.Module.SymbolReader;
            var method_debug_information = symbol_reader?.Read(method_definition);
            Collection<SequencePoint> sequence_points = method_debug_information != null ? method_debug_information.SequencePoints : new Collection<SequencePoint>();
            Mono.Cecil.Cil.MethodBody body = method_definition.Body;
            if (body == null)
                throw new Exception("Body null, not expecting it to be.");
            if (body.Instructions == null)
                throw new Exception("Body has instructions collection.");
            if (body.Instructions.Count == 0)
                throw new Exception("Body instruction count is zero.");
            SequencePoint previous_sp = null;
            for (int j = 0; j < instruction_count; ++j)
            {
                Mono.Cecil.Cil.Instruction instruction = body.Instructions[j];
                SequencePoint sp = sequence_points.Where(s => { return s.Offset == instruction.Offset; }).FirstOrDefault();
                if (sp == null) sp = previous_sp;
                INST wrapped_instruction = INST.Wrap(instruction, basic_block, sp);
                basic_block.Instructions.Add(wrapped_instruction);
                if (sp != null) previous_sp = sp;
            }

            var instructions_before_splits = basic_block.Instructions.ToList();

            // Accumulate targets of jumps. These are split points for block "v".
            // Accumalated splits are in "leader_list" following this for-loop.
            for (int j = 0; j < instruction_count; ++j)
            {
                INST wrapped_instruction = basic_block.Instructions[j];
                Mono.Cecil.Cil.OpCode opcode = wrapped_instruction.OpCode;
                Mono.Cecil.Cil.FlowControl flow_control = opcode.FlowControl;

                switch (flow_control)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                        {
                            object operand = wrapped_instruction.Operand;
                            // Two cases of branches:
                            // 1) operand is a single instruction;
                            // 2) operand is an array of instructions via a switch instruction.
                            // In doing this type casting, the resulting instructions are turned
                            // into def's, and operands no longer correspond to what was in the original
                            // method. We override the List<> compare to correct this problem.
                            Mono.Cecil.Cil.Instruction single_instruction = operand as Mono.Cecil.Cil.Instruction;
                            Mono.Cecil.Cil.Instruction[] array_of_instructions = operand as Mono.Cecil.Cil.Instruction[];
                            if (single_instruction != null)
                            {
                                INST sp = null;
                                for (int i = 0; i < basic_block.Instructions.Count(); ++i)
                                {
                                    if (basic_block.Instructions[i].Offset == single_instruction.Offset
                                        && basic_block.Instructions[i].OpCode == single_instruction.OpCode)
                                    {
                                        sp = basic_block.Instructions[i];
                                        break;
                                    }
                                }
                                if (sp == null) throw new Exception("Instruction go to not found.");
                                bool found = false;
                                foreach (INST split in split_point)
                                    if (split.Offset == sp.Offset && split.OpCode == sp.OpCode)
                                    {
                                        found = true;
                                        break;
                                    }
                                if (! found)
                                    split_point.Add(sp);
                            }
                            else if (array_of_instructions != null)
                            {
                                foreach (var ins in array_of_instructions)
                                {
                                    Debug.Assert(ins != null);
                                    INST sp = null;
                                    for (int i = 0; i < basic_block.Instructions.Count(); ++i)
                                    {
                                        if (basic_block.Instructions[i].Offset == ins.Offset
                                            && basic_block.Instructions[i].OpCode == ins.OpCode)
                                        {
                                            sp = basic_block.Instructions[i];
                                            break;
                                        }
                                    }
                                    if (sp == null) throw new Exception("Instruction go to not found.");
                                    bool found = false;
                                    foreach (INST split in split_point)
                                        if (split.Offset == sp.Offset && split.OpCode == sp.OpCode)
                                        {
                                            found = true;
                                            break;
                                        }
                                    if (!found)
                                        split_point.Add(sp);
                                }
                            }
                            else
                                throw new Exception("Unknown operand type for basic block partitioning.");
                        }
                        break;
                }

                // Split blocks after certain instructions, too.
                switch (flow_control)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Call:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                    case Mono.Cecil.Cil.FlowControl.Return:
                    case Mono.Cecil.Cil.FlowControl.Throw:
                        {
                            if (flow_control == Mono.Cecil.Cil.FlowControl.Call && ! Campy.Utils.Options.IsOn("split_at_calls"))
                                break;
                            if (j + 1 < instruction_count)
                            {
                                var ins = basic_block.Instructions[j + 1].Instruction;
                                INST sp = null;
                                for (int i = 0; i < basic_block.Instructions.Count(); ++i)
                                {
                                    if (basic_block.Instructions[i].Offset == ins.Offset
                                        && basic_block.Instructions[i].OpCode == ins.OpCode)
                                    {
                                        sp = basic_block.Instructions[i];
                                        break;
                                    }
                                }
                                if (sp == null) throw new Exception("Instruction go to not found.");
                                bool found = false;
                                foreach (INST split in split_point)
                                    if (split.Offset == sp.Offset && split.OpCode == sp.OpCode)
                                    {
                                        found = true;
                                        break;
                                    }
                                if (!found)
                                    split_point.Add(sp);
                            }
                        }
                        break;
                }
            }

            // Get try-catch blocks and add those split points.
            foreach (var eh in body.ExceptionHandlers)
            {
                var start = eh.TryStart;
                var end = eh.TryEnd;
                // Split at try start.
                Instruction ins = start;
                INST sp = null;
                for (int i = 0; i < basic_block.Instructions.Count(); ++i)
                {
                    if (basic_block.Instructions[i].Offset == ins.Offset
                        && basic_block.Instructions[i].OpCode == ins.OpCode)
                    {
                        sp = basic_block.Instructions[i];
                        break;
                    }
                }
                if (sp == null) throw new Exception("Instruction go to not found.");
                bool found = false;
                foreach (INST split in split_point)
                    if (split.Offset == sp.Offset && split.OpCode == sp.OpCode)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                    split_point.Add(sp);
            }

            // Note, we assume that these splits are within the same method.
            // Order the list according to offset from beginning of the method.
            List<INST> ordered_leader_list = new List<INST>();
            for (int j = 0; j < instruction_count; ++j)
            {
                // Order jump targets. These denote locations
                // where to split blocks. However, it's ordered,
                // so that splitting is done from last instruction in block
                // to first instruction in block.
                INST i = basic_block.Instructions[j];
                if (split_point.Contains(i,
                    new LambdaComparer<INST>(
                        (INST a, INST b)
                            =>
                        {
                            if (a.Offset != b.Offset)
                                return false;
                            if (a.OpCode != b.OpCode)
                                return false;
                            return true;
                        })))
                    ordered_leader_list.Add(i);
            }

            if (ordered_leader_list.Count != split_point.Count)
                throw new Exception(
                    "Mono Cecil giving weird results for instruction operand type casting. Size of original split points not the same as order list of split points.");
 
            // Split block at all jump targets.
            foreach (INST i in ordered_leader_list)
            {
                var owner = Cfg.PeekChangeSet(change_set).Where(
                    n => n.Instructions.Where(ins =>
                    {
                        if (ins.Offset != i.Offset)
                            return false;
                        if (ins.OpCode != i.OpCode)
                            return false;
                        return true;
                    }
                ).Any()).ToList();
                // Check if there are multiple nodes with the same instruction or if there isn't
                // any node found containing the instruction. Either way, it's a programming error.
                if (owner.Count != 1)
                    throw new Exception("Cannot find instruction!");
                CFG.Vertex target_node = owner.FirstOrDefault();
                var j = target_node.Instructions.FindIndex(a =>
                {
                    if (a.Offset != i.Offset)
                        return false;
                    if (a.OpCode != i.OpCode)
                        return false;
                    return true;
                });
                CFG.Vertex new_node = Split(target_node, j);
            }


            LambdaComparer<Instruction> fixed_comparer = new LambdaComparer<Instruction>(
                (Instruction a, Instruction b)
                    => a.Offset == b.Offset
                       && a.OpCode == b.OpCode
            );

            // Add in all edges.
            var list_new_nodes = Cfg.PopChangeSet(change_set);

            foreach (var node in list_new_nodes)
            {
                int node_instruction_count = node.Instructions.Count;
                INST last_instruction = node.Instructions[node_instruction_count - 1];

                Mono.Cecil.Cil.OpCode opcode = last_instruction.OpCode;
                Mono.Cecil.Cil.FlowControl flow_control = opcode.FlowControl;

                // Add jump edge.
                switch (flow_control)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                        {
                            // Two cases: i.Operand is a single instruction, or an array of instructions.
                            if (last_instruction.Operand as Mono.Cecil.Cil.Instruction != null)
                            {
                                // Handel leave instructions with code below.
                                if (!(last_instruction.OpCode.Code == Code.Leave ||
                                      last_instruction.OpCode.Code == Code.Leave_S))
                                {
                                    Mono.Cecil.Cil.Instruction target_instruction =
                                        last_instruction.Operand as Mono.Cecil.Cil.Instruction;
                                    CFG.Vertex target_node = list_new_nodes.FirstOrDefault(
                                        (CFG.Vertex x) =>
                                        {
                                            if (!fixed_comparer.Equals(x.Instructions.First().Instruction,
                                                target_instruction))
                                                return false;
                                            return true;
                                        });
                                    if (target_node != null)
                                        Cfg.AddEdge(new CFG.Edge() {From = node, To = target_node});
                                }
                            }
                            else if (last_instruction.Operand as Mono.Cecil.Cil.Instruction[] != null)
                            {
                                foreach (Mono.Cecil.Cil.Instruction target_instruction in
                                    (last_instruction.Operand as Mono.Cecil.Cil.Instruction[]))
                                {
                                    CFG.Vertex target_node = list_new_nodes.FirstOrDefault(
                                        (CFG.Vertex x) =>
                                        {
                                            if (!fixed_comparer.Equals(x.Instructions.First().Instruction, target_instruction))
                                                return false;
                                            return true;
                                        });
                                    if (target_node != null)
                                        Cfg.AddEdge(new CFG.Edge(){From = node, To = target_node});
                                }
                            }
                            else
                                throw new Exception("Unknown operand type for conditional branch.");
                            break;
                        }
                }

                // Add fall through edge.
                switch (flow_control)
                {
                    //case Mono.Cecil.Cil.FlowControl.Branch:
                    //case Mono.Cecil.Cil.FlowControl.Break:
                    case Mono.Cecil.Cil.FlowControl.Call:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                    case Mono.Cecil.Cil.FlowControl.Meta:
                    case Mono.Cecil.Cil.FlowControl.Next:
                    case Mono.Cecil.Cil.FlowControl.Phi:
                    case Mono.Cecil.Cil.FlowControl.Throw:
                        {
                            int next = instructions_before_splits.FindIndex(
                                           n =>
                                           {
                                               var r = n == last_instruction;
                                               return r;
                                           }
                                   );
                            if (next < 0)
                                break;
                            next += 1;
                            if (next >= instructions_before_splits.Count)
                                break;
                            var next_instruction = instructions_before_splits[next];
                            var owner = next_instruction.Block;
                            CFG.Vertex target_node = owner;
                            Cfg.AddEdge(new CFG.Edge(){From = node, To = target_node});
                        }
                        break;
                    case Mono.Cecil.Cil.FlowControl.Return:
                        if (last_instruction.Instruction.OpCode.Code == Code.Endfinally)
                        {
                            // Although the exception handling is like a procedure call,
                            // local variables are all accessible. So, we need to copy stack
                            // values around. In addition, we have to create a fall through
                            // even though there is stack unwinding.
                            int next = instructions_before_splits.FindIndex(
                                           n =>
                                           {
                                               var r = n == last_instruction;
                                               return r;
                                           }
                                   );
                            if (next < 0)
                                break;
                            next += 1;
                            if (next >= instructions_before_splits.Count)
                                break;
                            var next_instruction = instructions_before_splits[next];
                            var owner = next_instruction.Block;
                            CFG.Vertex target_node = owner;
                            Cfg.AddEdge(new CFG.Edge() { From = node, To = target_node });
                        }
                        break;
                }
            }

            // Get inclusive start/exclusive end ranges of try/catch/finally.
            Dictionary<int, int> exclusive_eh_range = new Dictionary<int, int>();
            foreach (var eh in body.ExceptionHandlers)
            {
                int try_start = eh.TryStart.Offset;
                int eh_end = eh.TryEnd != null ? eh.TryEnd.Offset : 0;
                if (eh.TryEnd != null && eh.TryEnd.Offset > eh_end)
                {
                    eh_end = eh.TryEnd.Offset;
                }
                if (eh.HandlerEnd != null && eh.HandlerEnd.Offset > eh_end)
                {
                    eh_end = eh.HandlerEnd.Offset;
                }
                exclusive_eh_range[try_start] = eh_end;
            }
            // Get inclusive start/inclusive end ranges of try/catch/finally.
            Dictionary<int, int> inclusive_eh_range = new Dictionary<int, int>();
            foreach (var pair in exclusive_eh_range)
            {
                int previous_instruction_address = 0;
                foreach (var i in body.Instructions)
                {
                    if (pair.Value == i.Offset)
                    {
                        inclusive_eh_range[pair.Key] = previous_instruction_address;
                        break;
                    }
                    previous_instruction_address = i.Offset;
                }
            }
            // Get "finally" blocks for each try, if there is one.
            Dictionary<int, CFG.Vertex> try_finally_block = new Dictionary<int, CFG.Vertex>();
            foreach (var eh in body.ExceptionHandlers)
            {
                if (eh.HandlerType == ExceptionHandlerType.Finally)
                {
                    var finally_entry_block = list_new_nodes.Where(
                        n =>
                        {
                            var first = n.Instructions.First().Instruction;
                            if (first.Offset == eh.HandlerStart.Offset)
                                return true;
                            else
                                return false;
                        }).First();
                    try_finally_block[eh.TryStart.Offset] = finally_entry_block;
                }
            }
            // Set block properties.
            foreach (var eh in body.ExceptionHandlers)
            {
                var block = list_new_nodes.Where(
                    n =>
                    {
                        var first = n.Instructions.First().Instruction;
                        if (first.Offset == eh.HandlerStart.Offset)
                            return true;
                        else
                            return false;
                    }).First();
                block.CatchType = eh.CatchType;
                block.IsCatch = eh.HandlerType == ExceptionHandlerType.Catch;
                block.ExceptionHandler = eh;
            }
            // Get "try" block for each try.
            Dictionary<int, CFG.Vertex> try_entry_block = new Dictionary<int, CFG.Vertex>();
            foreach (var pair in inclusive_eh_range)
            {
                int start = pair.Key;
                var entry_block = list_new_nodes.Where(
                    n =>
                    {
                        var first = n.Instructions.First().Instruction;
                        if (first.Offset == start)
                            return true;
                        else
                            return false;
                    }).First();
                try_entry_block[start] = entry_block;
            }
            // Get entry block for each exception handler.
            Dictionary<int, CFG.Vertex> eh_entry_block = new Dictionary<int, CFG.Vertex>();
            foreach (var eh in body.ExceptionHandlers)
            {
                int start = eh.HandlerStart.Offset;
                var entry_block = list_new_nodes.Where(
                    n =>
                    {
                        var first = n.Instructions.First().Instruction;
                        if (first.Offset == start)
                            return true;
                        else
                            return false;
                    }).First();
                eh_entry_block[start] = entry_block;
            }

            foreach (var eh in body.ExceptionHandlers)
            {
                int start = eh.TryStart.Offset;
                var try_block = try_entry_block[start];
                int eh_start = eh.HandlerStart.Offset;
                var eh_block = eh_entry_block[eh_start];
                if (eh.HandlerType == ExceptionHandlerType.Finally) continue;
                foreach (var prev in list_new_nodes.First()._graph.Predecessors(try_block))
                    Cfg.AddEdge(new CFG.Edge() { From = prev, To = eh_block });
            }

            // Go through all CIL "leave" instructions and draw up edges. Any leave to end of
            // endfinally block requires edge to finally block, not the following instruction.
            foreach (var node in list_new_nodes)
            {
                int node_instruction_count = node.Instructions.Count;
                INST leave_instruction = node.Instructions[node_instruction_count - 1];
                Mono.Cecil.Cil.OpCode opcode = leave_instruction.OpCode;
                Mono.Cecil.Cil.FlowControl flow_control = opcode.FlowControl;
                if (!(leave_instruction.OpCode.Code == Code.Leave || leave_instruction.OpCode.Code == Code.Leave_S))
                    continue;

                // Link up any leave instructions
                object operand = leave_instruction.Operand;
                Mono.Cecil.Cil.Instruction single_instruction = operand as Mono.Cecil.Cil.Instruction;
                Mono.Cecil.Cil.Instruction[] array_of_instructions = operand as Mono.Cecil.Cil.Instruction[];
                if (single_instruction == null) throw new Exception("Malformed leave instruction.");
                KeyValuePair<int, int> pair = inclusive_eh_range.Where(p => p.Key <= leave_instruction.Instruction.Offset
                                                         && leave_instruction.Instruction.Offset <= p.Value).FirstOrDefault();
                // pair indicates what try/catch/finally block. If not in a try/catch/finally,
                // draw edge to destination. If the destination is outside try/catch/finally,
                // draw edge to destination.
                if (pair.Value == 0 || single_instruction.Offset >= pair.Key && single_instruction.Offset <= pair.Value)
                {
                    var whereever = list_new_nodes.Where(
                        n =>
                        {
                            var first = n.Instructions.First().Instruction;
                            if (first.Offset == single_instruction.Offset)
                                return true;
                            else
                                return false;
                        }).First();
                    Cfg.AddEdge(new CFG.Edge() { From = node, To = whereever });
                    continue;
                }
                if (try_finally_block.ContainsKey(pair.Key))
                    Cfg.AddEdge(new CFG.Edge() { From = node, To = try_finally_block[pair.Key] });
                else
                {
                    var whereever = list_new_nodes.Where(
                        n =>
                        {
                            var first = n.Instructions.First().Instruction;
                            if (first.Offset == single_instruction.Offset)
                                return true;
                            else
                                return false;
                        }).First();
                    Cfg.AddEdge(new CFG.Edge() { From = node, To = whereever } );
                }
            }

            if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                Cfg.OutputDotGraph();
            if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                Cfg.OutputEntireGraph();
        }

        private CFG.Vertex Split(CFG.Vertex node, int i)
        {
            Debug.Assert(node.Instructions.Count != 0);
            // Split this node into two nodes, with all instructions after "i" in new node.
            var cfg = node._graph;
            CFG.Vertex result = (CFG.Vertex)cfg.AddVertex(new CFG.Vertex() { Name = cfg.NewNodeNumber().ToString() });
            result.Entry = node.Entry;
            result._method_definition = node._method_definition;
            result._method_reference = node._method_reference;

            int count = node.Instructions.Count;

            if (Campy.Utils.Options.IsOn("cfg_construction_trace"))
            {
                System.Console.WriteLine("Split node " + node.Name + " at instruction " + node.Instructions[i].Instruction);
                System.Console.WriteLine("Node prior to split:");
                node.OutputEntireNode();
                System.Console.WriteLine("New node is " + result.Name);
            }

            if (!result._method_reference.Module.HasSymbols)
            {
                // Try to get symbols, but if none available, don't worry about it.
                try { result._method_reference.Module.ReadSymbols(); } catch { }
            }
            var symbol_reader = result._method_reference.Module.SymbolReader;
            var method_debug_information = symbol_reader?.Read(result._method_definition);
            Collection<SequencePoint> sequence_points = method_debug_information != null ? method_debug_information.SequencePoints : new Collection<SequencePoint>();

            // Add instructions from split point to new block, including any debugging information.
            for (int j = i; j < count; ++j)
            {
                var offset = node.Instructions[j].Instruction.Offset;
                // Do not re-wrap the instruction, simply move wrapped instructions.
                INST old_inst = node.Instructions[j];
                result.Instructions.Add(old_inst);
                // Correct Block to point to new block.
                old_inst.Block = result;
            }

            // Remove instructions from this block.
            for (int j = i; j < count; ++j)
            {
                node.Instructions.RemoveAt(i);
            }

            Debug.Assert(node.Instructions.Count != 0);
            Debug.Assert(result.Instructions.Count != 0);
            Debug.Assert(node.Instructions.Count + result.Instructions.Count == count);

            INST last_instruction = node.Instructions[node.Instructions.Count - 1];

            // Transfer any out edges to pred block to new block.
            while (cfg.SuccessorNodes(node).Count() > 0)
            {
                CFG.Vertex succ = cfg.SuccessorNodes(node).First();
                cfg.DeleteEdge(new CFG.Edge() { From = node, To = succ });
                cfg.AddEdge(new CFG.Edge() { From = result, To = succ });
            }

            if (Campy.Utils.Options.IsOn("cfg_construction_trace"))
            {
                System.Console.WriteLine("Node after split:");
                node.OutputEntireNode();
                System.Console.WriteLine("Newly created node:");
                result.OutputEntireNode();
                System.Console.WriteLine();
            }

            return result;
        }
    }
}
