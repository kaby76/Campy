
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

    public class IMPORTER
    {
        private StackQueue<Tuple<MethodReference, List<TypeReference>>> _methods_to_do;
        private List<string> _methods_done;

        public bool Failed
        {
            get; internal set;
        }

        public CFG Cfg
        {
            get; internal set;
        }

        public IMPORTER()
        {
            Cfg = new CFG();
            _methods_to_do = new StackQueue<Tuple<MethodReference, List<TypeReference>>>();
            _methods_done = new List<string>();
        }

        public void AnalyzeMethod(MethodInfo methodInfo)
        {
            Failed = false; // Might be dubious to reset here.
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
            Cfg.OutputEntireGraph();
            Cfg.OutputDotGraph();
        }

        public void Add(MethodInfo method_info)
        {
            String kernel_assembly_file_name = method_info.DeclaringType.Assembly.Location;
            string p = Path.GetDirectoryName(kernel_assembly_file_name);
            var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(p);
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(
                kernel_assembly_file_name,
                new ReaderParameters { AssemblyResolver = resolver, ReadSymbols = true});
            MethodReference method_reference = md.Import(method_info);
            Add(method_reference);
        }

        public void Add(MethodReference method_reference)
        {
            // Do not analyze if nonsense.
            if (method_reference == null) return;
            // Do not analyze if already being or has been considered.
            if (Cfg.MethodAvoid(method_reference.FullName)) return;
            if (_methods_done.Contains(method_reference.FullName)) return;
            foreach (var tuple in _methods_to_do)
            {
                if (tuple.Item1.FullName == method_reference.FullName) return;
            }
            _methods_to_do.Push(new Tuple<MethodReference, List<TypeReference>>(method_reference, new List<TypeReference>()));
        }

        public void ExtractBasicBlocks()
        {
            while (_methods_to_do.Count > 0)
            {
                int change_set_id = this.Cfg.StartChangeSet();

                Tuple<MethodReference, List<TypeReference>> definition = _methods_to_do.Pop();

                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("ExtractBasicBlocks for " + definition.Item1.FullName);

                ExtractBasicBlocksOfMethod(definition);

                var blocks = this.Cfg.PopChangeSet(change_set_id);

                // Get closure of calls, if possible.
                foreach (var b in blocks)
                {
                    foreach (INST i in b.Instructions)
                    {
                        var fc = i.OpCode.FlowControl;
                        if (fc != Mono.Cecil.Cil.FlowControl.Call)
                            continue;
                        object method = i.Operand;
                        if (method as MethodReference != null)
                        {
                            MethodReference mr = method as MethodReference;
                            Add(mr);
                        }
                    }
                }
            }
        }

        private static MethodInfo GetMethodInfo(Action a)
        {
            return a.Method;
        }

        public static ModuleDefinition LoadAssembly(String assembly_file_name)
        {
            // Microsoft keeps screwing and changing the code that finds assemblies all the damn time.
            // Get the type of Importer. Assume that it's in the same directory as everything else.
            // Set that path for the god damn resolver.
            var fucking_directory_path = Campy.Utils.CampyInfo.PathOfCampy();
            var full_frigging_path_of_assembly_file = fucking_directory_path + "\\" + assembly_file_name;
            var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(fucking_directory_path);
            if (File.Exists(full_frigging_path_of_assembly_file))
                assembly_file_name = full_frigging_path_of_assembly_file;
            ModuleDefinition module = ModuleDefinition.ReadModule(
                assembly_file_name,
                new ReaderParameters { AssemblyResolver = resolver, ReadSymbols = true});
            return module;
        }

        static MethodReference MakeGeneric(MethodReference method, TypeReference declaringType)
        {
            var reference = new MethodReference(method.Name, method.ReturnType, declaringType);

            foreach (ParameterDefinition parameter in method.Parameters)
                reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));
            return reference;
        }

        static TypeReference MakeGenericType(TypeReference type, params
            TypeReference[] arguments)
        {
            if (type.GenericParameters.Count != arguments.Length)
                throw new ArgumentException();

            var instance = new GenericInstanceType(type);
            foreach (var argument in arguments)
                instance.GenericArguments.Add(argument);

            return instance;
        }

        public static MethodReference MakeHostInstanceGeneric(
            MethodReference self,
            params TypeReference[] args)
        {
            var reference = new MethodReference(
                self.Name,
                self.ReturnType,
                self.DeclaringType.MakeGenericInstanceType(args))
            {
                HasThis = self.HasThis,
                ExplicitThis = self.ExplicitThis,
                CallingConvention = self.CallingConvention
            };

            foreach (var parameter in self.Parameters)
                reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));

            foreach (var genericParam in self.GenericParameters)
                reference.GenericParameters.Add(new GenericParameter(genericParam.Name, reference));

            return reference;
        }

        public static MethodReference MakeHostInstanceGeneric2(
            GenericInstanceType declaring_type,
            MethodReference self
            )
        {
            var reference = new MethodReference(
                self.Name,
                self.ReturnType,
                declaring_type)
            {
                HasThis = self.HasThis,
                ExplicitThis = self.ExplicitThis,
                CallingConvention = self.CallingConvention
            };

            foreach (ParameterDefinition parameter in self.Parameters)
            {
                TypeReference type_reference_of_parameter = parameter.ParameterType;

                Collection<TypeReference> gp = declaring_type.GenericArguments;
                // Map parameter to actual type.
                if (type_reference_of_parameter.IsGenericParameter)
                {
                    // Get arg number.
                    int num = Int32.Parse(type_reference_of_parameter.Name.Substring(1));
                    var yo = gp.ToArray()[num];
                    type_reference_of_parameter = yo;
                }
                reference.Parameters.Add(new ParameterDefinition(type_reference_of_parameter));
            }

            foreach (var genericParam in self.GenericParameters)
                reference.GenericParameters.Add(new GenericParameter(genericParam.Name, reference));

            return reference;
        }

        private void ExtractBasicBlocksOfMethod(Tuple<MethodReference, List<TypeReference>> definition)
        {
            MethodReference original_method_reference = definition.Item1;
            _methods_done.Add(original_method_reference.FullName);
            MethodDefinition method_definition = Rewrite(original_method_reference);
            if (method_definition == null)
                return;
            _methods_done.Add(method_definition.FullName);
            int change_set = Cfg.StartChangeSet();
            int instruction_count = method_definition.Body.Instructions.Count;
            List<Mono.Cecil.Cil.Instruction> split_point = new List<Mono.Cecil.Cil.Instruction>();
            // Each method is a leader of a block.
            CFG.Vertex basic_block = (CFG.Vertex)Cfg.AddVertex(new CFG.Vertex(){Name = Cfg.NewNodeNumber().ToString()});
            basic_block._method_definition = method_definition;
            basic_block._original_method_reference = original_method_reference;
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
            for (int j = 0; j < instruction_count; ++j)
            {
                Mono.Cecil.Cil.Instruction instruction = method_definition.Body.Instructions[j];
                INST wrapped_instruction = INST.Wrap(instruction, basic_block, sequence_points.Where(sp => { return sp.Offset == instruction.Offset; }).FirstOrDefault());
                basic_block.Instructions.Add(wrapped_instruction);
            }

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
                            Mono.Cecil.Cil.Instruction single_instruction = operand as Mono.Cecil.Cil.Instruction;
                            Mono.Cecil.Cil.Instruction[] array_of_instructions = operand as Mono.Cecil.Cil.Instruction[];
                            if (single_instruction != null)
                            {
                                if (!split_point.Contains(single_instruction))
                                    split_point.Add(single_instruction);
                            }
                            else if (array_of_instructions != null)
                            {
                                foreach (var ins in array_of_instructions)
                                {
                                    Debug.Assert(ins != null);
                                    if (!split_point.Contains(single_instruction))
                                        split_point.Add(ins);
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
                                if (!split_point.Contains(ins))
                                    split_point.Add(ins);
                            }
                        }
                        break;
                }
            }

            // Note, we assume that these splits are within the same method.
            // Order the list according to offset from beginning of the method.
            List<Mono.Cecil.Cil.Instruction> ordered_leader_list = new List<Mono.Cecil.Cil.Instruction>();
            for (int j = 0; j < instruction_count; ++j)
            {
                // Order jump targets. These denote locations
                // where to split blocks. However, it's ordered,
                // so that splitting is done from last instruction in block
                // to first instruction in block.
                Mono.Cecil.Cil.Instruction i = method_definition.Body.Instructions[j];
                if (split_point.Contains(i))
                    ordered_leader_list.Add(i);
            }

            // Split block at all jump targets.
            foreach (var i in ordered_leader_list)
            {
                var owner = Cfg.Vertices.Where(
                    n => n.Instructions.Where(ins => ins.Instruction == i).Any()).ToList();
                // Check if there are multiple nodes with the same instruction or if there isn't
                // any node found containing the instruction. Either way, it's a programming error.
                if (owner.Count != 1)
                    throw new Exception("Cannot find instruction!");
                CFG.Vertex target_node = owner.FirstOrDefault();
                var j = target_node.Instructions.FindIndex(a => a.Instruction == i);
                CFG.Vertex new_node = target_node.Split(j);
            }

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
                                Mono.Cecil.Cil.Instruction target_instruction =
                                    last_instruction.Operand as Mono.Cecil.Cil.Instruction;
                                CFG.Vertex target_node = Cfg.Vertices.First(
                                    (CFG.Vertex x) =>
                                    {
                                        if (!x.Instructions.First().Instruction.Equals(target_instruction))
                                            return false;
                                        return true;
                                    });
                                Cfg.AddEdge(new CFG.Edge(){From = node, To = target_node});
                            }
                            else if (last_instruction.Operand as Mono.Cecil.Cil.Instruction[] != null)
                            {
                                foreach (Mono.Cecil.Cil.Instruction target_instruction in
                                    (last_instruction.Operand as Mono.Cecil.Cil.Instruction[]))
                                {
                                    CFG.Vertex target_node = Cfg.Vertices.First(
                                        (CFG.Vertex x) =>
                                        {
                                            if (!x.Instructions.First().Instruction.Equals(target_instruction))
                                                return false;
                                            return true;
                                        });
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
                    //case Mono.Cecil.Cil.FlowControl.Return:
                    case Mono.Cecil.Cil.FlowControl.Throw:
                        {
                            int next = method_definition.Body.Instructions.ToList().FindIndex(
                                           n =>
                                           {
                                               var r = n == last_instruction.Instruction &&
                                                       n.Offset == last_instruction.Instruction.Offset
                                                   ;
                                               return r;
                                           }
                                   );
                            if (next < 0)
                                break;
                            next += 1;
                            if (next >= method_definition.Body.Instructions.Count)
                                break;
                            var next_instruction = method_definition.Body.Instructions[next];
                            var owner = Cfg.Vertices.Where(
                                n => n.Instructions.Where(ins => ins.Instruction == next_instruction).Any()).ToList();
                            if (owner.Count != 1)
                                throw new Exception("Cannot find instruction!");
                            CFG.Vertex target_node = owner.FirstOrDefault();
                            Cfg.AddEdge(new CFG.Edge(){From = node, To = target_node});
                        }
                        break;
                }
            }
            Cfg.OutputDotGraph();
            Cfg.OutputEntireGraph();
        }

        /// <summary>
        /// Substitute method reference with an method reference in the BCL runtime for GPUs if there is one.
        /// The substituted method is returned. If there is no substitution, it will return the method reference unchanged.
        /// In the body of the method, rewrite each instruction, replacing references to the BCL runtime for GPUs
        /// </summary>
        /// <param name="method_reference"></param>
        /// <returns></returns>
        public MethodDefinition Rewrite(MethodReference method_reference)
        {
            var method_definition = RUNTIME.SubstituteMethod(method_reference);

            if (!(method_definition == null || method_definition.Body == null))
                RUNTIME.RewriteCilCodeBlock(method_definition.Body);

            return method_definition;
        }
    }
}
