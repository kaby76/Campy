
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
        private CFG _cfg;
        private List<ModuleDefinition> _loaded_modules;
        private List<ModuleDefinition> _analyzed_modules;

        // After studying this for a while, I've come to the conclusion that decompiling methods
        // requires type information, as methods/"this" could be generic. So, we create a list of
        // Tuple<MethodDefition, List<TypeReference>> that indicates the method, and generic parameters.
        private StackQueue<Tuple<MethodReference, List<TypeReference>>> _methods_to_do;
        private List<string> _methods_done;

        // Some methods references resolve to null. And, some methods we might want to substitute a
        // different implementation that the one normally found through reference Resolve(). Retain a
        // mapping of methods to be rewritten.
        private Dictionary<string, string> _rewritten_runtime;

        public CFG Cfg
        {
            get { return _cfg; }
            set { _cfg = value; }
        }

        public Reader()
        {
            _cfg = new CFG();
            _loaded_modules = new List<ModuleDefinition>();
            _analyzed_modules = new List<ModuleDefinition>();
            _methods_to_do = new StackQueue<Tuple<MethodReference, List<TypeReference>>>();
            _methods_done = new List<string>();
            _rewritten_runtime = new Dictionary<string, string>();
            _rewritten_runtime.Add("System.Int32 System.Int32[0...,0...]::Get(System.Int32,System.Int32)",
                "T Campy.Compiler.Runtime`1::get_multi_array(System.Array,System.Int32,System.Int32)");
            _rewritten_runtime.Add("System.Void System.Int32[0...,0...]::Set(System.Int32,System.Int32,System.Int32)",
                "System.Void Campy.Compiler.Runtime`1::set_multi_array(System.Array,System.Int32,System.Int32,T)");
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

        public void AnalyzeMethod(MethodInfo methodInfo)
        {
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
            _cfg.OutputEntireGraph();
            _cfg.OutputDotGraph();
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
            String kernel_assembly_file_name = reference.DeclaringType.Assembly.Location;
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(kernel_assembly_file_name);
            MethodReference refer = md.Import(reference);
            Add(refer);
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

        public void Add(MethodReference method_reference)
        {
            if (method_reference == null)
                return;
            if (_cfg.MethodAvoid(method_reference.FullName))
                return;
            if (_methods_done.Contains(method_reference.FullName))
                return;
            foreach (var tuple in _methods_to_do)
            {
                if (tuple.Item1.FullName == method_reference.FullName)
                    return;
            }
            _methods_to_do.Push(new Tuple<MethodReference, List<TypeReference>>(method_reference, new List<TypeReference>()));
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

        private MethodDefinition GetDefinition(string name)
        {
            int index_of_colons = name.IndexOf("::");
            int index_of_start_namespace = name.Substring(0, index_of_colons).LastIndexOf(" ") + 1;
            string ns = name.Substring(index_of_start_namespace, index_of_colons - index_of_start_namespace);
            ns = ns.Substring(0, ns.LastIndexOf("."));
            ns = ns + ".dll";
            var result2 = LoadAssembly(ns);
            foreach (var type in result2.Types)
            {
                System.Console.WriteLine(type.FullName);
                foreach (var method in type.Methods)
                {
                    if (method.FullName == name)
                        return method;
                }
            }
            return null;
        }

        private void ExtractBasicBlocksOfMethod(Tuple<MethodReference, List<TypeReference>> definition)
        {
            MethodReference item1 = definition.Item1;
            List<TypeReference> item2 = definition.Item2;

            _methods_done.Add(item1.FullName);

            // Make sure definition assembly is loaded. The analysis of the method cannot
            // be done if the routine hasn't been loaded into Mono!
            String full_name = item1.Module.FullyQualifiedName;
            var result = LoadAssembly(full_name);

            var git = item1.DeclaringType as GenericInstanceType;
            if (git != null)
            {
                var xx = MakeHostInstanceGeneric2(git, item1);
                // We can rewrite the method to not contain generic parameters, but it ends up
                // with Resolve() returning null.
            }

            // Resolve() tends to turn anything into mush. It removes type information
            // per instruction. Set up for analysis of the method body, if there is one.
            MethodDefinition md = item1.Resolve();

            item1.Rewrite();

            Mono.Cecil.Cil.MethodBody body = null;
            bool has_ret = false;
            if (md == null)
            {
                // Note, some situations MethodDefinition.Resolve() return null.
                // For a good introduction on Resolve(), see https://github.com/jbevain/cecil/wiki/Resolving
                // According to that, it should always return non-null. However, multi-dimensional arrays do not
                // seem to resolve. Moreover, to add more confusion, a one dimensional array do resolve,
                // As arrays are so central to GPU programming, we need to substitute for System.Array code--that
                // we cannot find--into code from a runtime library.
                System.Console.WriteLine("No definition for " + item1.FullName);
                System.Console.WriteLine(item1.IsDefinition ? "" : "Is not a definition at that!");
                var name = item1.FullName;
                int[,] ar = new int[1, 1];
                var found = _rewritten_runtime.ContainsKey(name);
                if (found)
                {
                    string rewrite = _rewritten_runtime[name];
                    var def = GetDefinition(rewrite);
                    body = def.Body;
                }
                if (body == null)
                    return;
            }
            else if (md.Body == null)
            {
                System.Console.WriteLine("WARNING: METHOD BODY NULL! " + definition);
                return;
            }
            else
            {
                has_ret =  md.IsReuseSlot;
                body = md.Body;
            }
            int instruction_count = body.Instructions.Count;
            StackQueue<Mono.Cecil.Cil.Instruction> leader_list = new StackQueue<Mono.Cecil.Cil.Instruction>();

            // Each method is a leader of a block.
            CFG.Vertex v = (CFG.Vertex)_cfg.AddVertex(_cfg.NewNodeNumber());
            v.Method = item1;
            v.HasReturnValue = has_ret;
            v.Entry = v;
            _cfg.Entries.Add(v);
            for (int j = 0; j < instruction_count; ++j)
            {
                // accumulate jump to locations since these split blocks.

                // NB: This gets generic code. We have to instantiate it.

                Mono.Cecil.Cil.Instruction mi = body.Instructions[j];
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
                Mono.Cecil.Cil.Instruction i = body.Instructions[j];
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
