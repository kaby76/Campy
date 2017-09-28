
namespace Campy.Compiler
{
    using Campy.Utils;
    using Mono.Cecil;
    using Mono.Cecil.Rocks;
    using Mono.Collections.Generic;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Reflection;

    public class Reader
    {
        private CFG _cfg = new CFG();
        private List<ModuleDefinition> _loaded_modules = new List<ModuleDefinition>();
        private List<ModuleDefinition> _analyzed_modules = new List<ModuleDefinition>();

        // After studying this for a while, I've come to the conclusion that decompiling methods
        // requires type information, as methods/"this" could be generic. So, we create a list of
        // Tuple<MethodDefition, List<TypeReference>> that indicates the method, and generic parameters.
        private StackQueue<Tuple<MethodReference, List<TypeReference>>> _methods_to_do = new StackQueue<Tuple<MethodReference, List<TypeReference>>>();
        private List<string> _methods_done = new List<string>(); // No longer MethodDefition because there is no equivalence.

        public CFG Cfg
        {
            get { return _cfg; }
            set { _cfg = value; }
        }

        public void AnalyzeThisAssembly()
        {
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
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
            _cfg.OutputEntireGraph();
            _cfg.OutputDotGraph();
        }

        public void AnalyzeMethod(MethodInfo methodInfo)
        {
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
        }

        public void AnalyzeMethod(MethodDefinition md)
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
            MethodDefinition definition = reference.ToMonoMethodDefinition();
            Add(definition);
        }

        public void AddAssembly(String assembly_file_name)
        {
            ModuleDefinition module = LoadAssembly(assembly_file_name);
            String full_name = System.IO.Path.GetFullPath(assembly_file_name);
            foreach (ModuleDefinition md in this._analyzed_modules)
                if (md.FullyQualifiedName.Equals(full_name))
                    return;
            _analyzed_modules.Add(module);
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

        public void Add(Mono.Cecil.TypeReference type)
        {
            TypeDefinition type_defintion = type.Resolve();
            foreach (MethodDefinition definition in type_defintion.Methods)
                Add(definition);
        }

        public void Add(MethodReference definition)
        {
            if (definition == null)
                return;
            if (_cfg.MethodAvoid(definition.FullName))
                return;
            if (_methods_done.Contains(definition.FullName))
                return;

            // Get instantiated version of method if generic.
            var generic = definition.HasGenericParameters;
            var is_instance = definition.IsGenericInstance;
            var declaring_type = definition.DeclaringType;
            if (declaring_type != null)
            {
                var dt_generic_instance = declaring_type.IsGenericInstance;
                var dt_generic = declaring_type.HasGenericParameters;
            }

            foreach (var tuple in _methods_to_do)
            {
                if (tuple.Item1.FullName == definition.FullName)
                {
                    return;
                }
            }
            _methods_to_do.Push(new Tuple<MethodReference, List<TypeReference>>(definition, new List<TypeReference>()));
        }

        public void AddAssembly(Assembly assembly)
        {
            String assembly_file_name = assembly.Location;
            AddAssembly(assembly_file_name);
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
                    foreach (Inst i in b.Instructions)
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

        private ModuleDefinition LoadAssembly(String assembly_file_name)
        {
            String full_name = System.IO.Path.GetFullPath(assembly_file_name);
            foreach (ModuleDefinition md in this._loaded_modules)
                if (md.FullyQualifiedName.Equals(full_name))
                    return md;
            ModuleDefinition module = ModuleDefinition.ReadModule(assembly_file_name);
            _loaded_modules.Add(module);
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
            MethodReference item1 = definition.Item1;
            List<TypeReference> item2 = definition.Item2;

            _methods_done.Add(item1.FullName);

            // Make sure definition assembly is loaded. The analysis of the method cannot
            // be done if the routine hasn't been loaded into Mono!
            String full_name = item1.Module.FullyQualifiedName;
            LoadAssembly(full_name);

            var git = item1.DeclaringType as GenericInstanceType;
            if (git != null)
            {
                var xx = MakeHostInstanceGeneric2(git, item1);
                // We can rewrite the method to not contain generic parameters, but it ends up
                // with Resolve() returning null.
            }

            // Resolve() tends to turn anything into mush. It removes type information
            // per instruction. Use as a last resort!
            MethodDefinition md = item1.Resolve();
            if (md.Body == null)
            {
                throw new Exception("WARNING: METHOD BODY NULL! " + definition);
            }
            int instruction_count = md.Body.Instructions.Count;
            StackQueue<Mono.Cecil.Cil.Instruction> leader_list = new StackQueue<Mono.Cecil.Cil.Instruction>();

            // Each method is a leader of a block.
            CFG.Vertex v = (CFG.Vertex)_cfg.AddVertex(_cfg.NewNodeNumber());
            v.Method = item1;
            v.HasReturnValue = item1.Resolve().IsReuseSlot;
            v.Entry = v;
            _cfg.Entries.Add(v);
            for (int j = 0; j < instruction_count; ++j)
            {
                // accumulate jump to locations since these split blocks.

                // NB: This gets generic code. We have to instantiate it.

                Mono.Cecil.Cil.Instruction mi = item1.Resolve().Body.Instructions[j];
                //System.Console.WriteLine(mi);
                Inst i = Inst.Wrap(mi);
                i.Block = v;
                Mono.Cecil.Cil.OpCode op = i.OpCode;
                Mono.Cecil.Cil.FlowControl fc = op.FlowControl;

                v.Instructions.Add(i);

                if (fc == Mono.Cecil.Cil.FlowControl.Branch || fc == Mono.Cecil.Cil.FlowControl.Cond_Branch)
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

            // Accumalated splits are in "leader_list". These splits must be within the smae method.
            StackQueue<int> ordered_leader_list = new StackQueue<int>();
            for (int j = 0; j < instruction_count; ++j)
            {
                // Order jump targets. These denote locations
                // where to split blocks. However, it's ordered,
                // so that splitting is done from last instruction in block
                // to first instruction in block.
                Mono.Cecil.Cil.Instruction i = md.Body.Instructions[j];
                if (leader_list.Contains(i))
                    ordered_leader_list.Push(j);
            }

            // Split block at jump targets in reverse.
            while (ordered_leader_list.Count > 0)
            {
                int i = ordered_leader_list.Pop();
                CFG.Vertex new_node = v.Split(i);
            }

            StackQueue<CFG.Vertex> stack = new StackQueue<CFG.Vertex>();
            foreach (CFG.Vertex node in _cfg.VertexNodes) stack.Push(node);
            while (stack.Count > 0)
            {
                // Split blocks at branches. Set following instruction a leader
                // of the new block.
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
                        if (!Campy.Utils.Options.IsOn("split_at_calls"))
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
                            // By default, we no longer split at calls. Splitting causes
                            // problems because interprocedural edges are
                            // produced. That's not good because it makes
                            // code too "messy".
                            if (Campy.Utils.Options.IsOn("split_at_calls"))
                            {
                                object o = i.Operand;
                                if (o as MethodReference != null)
                                {
                                    MethodReference r = o as MethodReference;
                                    MethodDefinition d = r.Resolve();
                                    IEnumerable<CFG.Vertex> target_node_list = _cfg.VertexNodes.Where(
                                        (CFG.Vertex x) =>
                                        {
                                            return x.Method.FullName == r.FullName
                                                   && x.Entry == x;
                                        });
                                    int c = target_node_list.Count();
                                    if (c >= 1)
                                    {
                                        // target_node is the entry for a method. Also get the exit.
                                        CFG.Vertex target_node = target_node_list.First();
                                        CFG.Vertex exit_node = target_node.Exit;
                                    }
                                }
                            }
                            break;
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
