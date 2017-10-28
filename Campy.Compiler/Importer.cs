
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
    using Mono.Cecil.Cil;

    public class Importer
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

        public bool Failed
        {
            get;
            internal set;
        }

        public CFG Cfg
        {
            get { return _cfg; }
            set { _cfg = value; }
        }

        public Importer()
        {
            _cfg = new CFG();
            _loaded_modules = new List<ModuleDefinition>();
            _analyzed_modules = new List<ModuleDefinition>();
            _methods_to_do = new StackQueue<Tuple<MethodReference, List<TypeReference>>>();
            _methods_done = new List<string>();
            _rewritten_runtime = new Dictionary<string, string>();
            _rewritten_runtime.Add("System.Int32 System.Int32[0...,0...]::Get(System.Int32,System.Int32)",
                "System.Int32 Campy.Compiler.Runtime::get_multi_array(Campy.Compiler.Runtime/A*,System.Int32,System.Int32)");
            _rewritten_runtime.Add("System.Void System.Int32[0...,0...]::Set(System.Int32,System.Int32,System.Int32)",
                "System.Void Campy.Compiler.Runtime::set_multi_array(Campy.Compiler.Runtime/A*,System.Int32,System.Int32,System.Int32)");
            _rewritten_runtime.Add("System.Int32 System.Array::GetLength(System.Int32)",
                "System.Int32 Campy.Compiler.Runtime::get_length_multi_array(Campy.Compiler.Runtime/A*,System.Int32)");
            _rewritten_runtime.Add("System.Double System.Math::Sin(System.Double)",
                "System.Double Campy.Compiler.Runtime::Sine(System.Double)");
            _rewritten_runtime.Add("System.Double System.Math::Cos(System.Double)",
                "System.Double Campy.Compiler.Runtime::Cosine(System.Double)");
            _rewritten_runtime.Add("System.Double System.Math::Abs(System.Double)",
                "System.Double Campy.Compiler.Runtime::Abs(System.Double)");
            _rewritten_runtime.Add("System.Void System.ThrowHelper::ThrowArgumentOutOfRangeException()",
                "System.Void Campy.Compiler.Runtime::ThrowArgumentOutOfRangeException()");
            _rewritten_runtime.Add("System.Void System.Int32[0...,0...,0...]::Set(System.Int32,System.Int32,System.Int32,System.Int32)",
                "System.Void Campy.Compiler.Runtime::set_multi_array(Campy.Compiler.Runtime/A*,System.Int32,System.Int32,System.Int32,System.Int32)");
        }

        public void AnalyzeMethod(MethodInfo methodInfo)
        {
            Failed = false; // Might be dubious to reset here.
            this.Add(methodInfo);
            this.ExtractBasicBlocks();
            _cfg.OutputEntireGraph();
            _cfg.OutputDotGraph();
        }

        public void Add(MethodInfo method_info)
        {
            String kernel_assembly_file_name = method_info.DeclaringType.Assembly.Location;
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(kernel_assembly_file_name);
            MethodReference method_reference = md.Import(method_info);
            Add(method_reference);
        }

        public void Add(MethodReference method_reference)
        {
            // Do not analyze if nonsense.
            if (method_reference == null) return;
            // Do not analyze if already being or has been considered.
            if (_cfg.MethodAvoid(method_reference.FullName)) return;
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
            MethodReference original_method_reference = definition.Item1;
            List<TypeReference> item2 = definition.Item2;
            String full_name = original_method_reference.Module.FullyQualifiedName;
            var result = LoadAssembly(full_name);

            _methods_done.Add(original_method_reference.FullName);

            // Resolve() tends to turn anything into mush. It removes type information
            // per instruction. Set up for analysis for resolved method reference or a substituted
            // method reforence for inscrutible NET runtime.

            MethodDefinition method_definition = Rewrite(original_method_reference);
            if (method_definition == null) return;

            _methods_done.Add(method_definition.FullName);

            // Rewrite all instructions in block so as it makes sense.
            // In particular, replace references to runtime with instructions in Campy runtime.
            // Some NET runtime cannot be discovered (e.g., multidimensional get and set, and Math
            // methods).
            RewriteCilCodeBlock(method_definition.Body);

            int change_set = _cfg.StartChangeSet();

            int instruction_count = method_definition.Body.Instructions.Count;
            List<Mono.Cecil.Cil.Instruction> split_point = new List<Mono.Cecil.Cil.Instruction>();

            // Each method is a leader of a block.
            CFG.Vertex v = (CFG.Vertex)_cfg.AddVertex(_cfg.NewNodeNumber());

            Mono.Cecil.MethodReturnType rt = method_definition.MethodReturnType;
            Mono.Cecil.TypeReference tr = rt.ReturnType;
            var ret = tr.FullName != "System.Void";
            v.HasScalarReturnValue = ret && ! tr.IsStruct();
            v.HasStructReturnValue = ret && tr.IsStruct();
            v.ExpectedCalleeSignature = original_method_reference;
            v.RewrittenCalleeSignature = method_definition;
            v.HasThis = method_definition.HasThis;
            v.NumberOfArguments = method_definition.Parameters.Count
                + (v.HasThis ? 1 : 0)
                + (v.HasStructReturnValue ? 1 : 0);
            Mono.Cecil.MethodReference mr = v.RewrittenCalleeSignature;
            int locals = mr.Resolve().Body.Variables.Count;
            v.NumberOfLocals = locals;

            v.Entry = v;
            _cfg.Entries.Add(v);

            // Add instructions to the basic block.
            for (int j = 0; j < instruction_count; ++j)
            {
                Mono.Cecil.Cil.Instruction mi = method_definition.Body.Instructions[j];
                Inst i = Inst.Wrap(mi);
                i.Block = v;
                Mono.Cecil.Cil.OpCode op = i.OpCode;
                Mono.Cecil.Cil.FlowControl fc = op.FlowControl;
                v.Instructions.Add(i);
            }

            // Accumulate targets of jumps. These are split points for block "v".
            // Accumalated splits are in "leader_list" following this for-loop.
            for (int j = 0; j < instruction_count; ++j)
            {
                Inst i = v.Instructions[j];
                Mono.Cecil.Cil.OpCode op = i.OpCode;
                Mono.Cecil.Cil.FlowControl fc = op.FlowControl;

                switch (fc)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                        {
                            object o = i.Operand;
                            // Two cases of branches:
                            // 1) operand is a single instruction;
                            // 2) operand is an array of instructions via a switch instruction.
                            Mono.Cecil.Cil.Instruction single_instruction = o as Mono.Cecil.Cil.Instruction;
                            Mono.Cecil.Cil.Instruction[] list_of_instructions = o as Mono.Cecil.Cil.Instruction[];
                            if (single_instruction != null)
                            {
                                if (!split_point.Contains(single_instruction))
                                    split_point.Add(single_instruction);
                            }
                            else if (list_of_instructions != null)
                            {
                                foreach (var ins in list_of_instructions)
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
                switch (fc)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Call:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                    case Mono.Cecil.Cil.FlowControl.Return:
                    case Mono.Cecil.Cil.FlowControl.Throw:
                        {
                            if (fc == Mono.Cecil.Cil.FlowControl.Call && ! Campy.Utils.Options.IsOn("split_at_calls"))
                                break;
                            if (j + 1 < instruction_count)
                            {
                                var ins = v.Instructions[j + 1].Instruction;
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
                var owner = _cfg.VertexNodes.Where(
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
            var list_new_nodes = _cfg.PopChangeSet(change_set);
            foreach (var node in list_new_nodes)
            {
                int node_instruction_count = node.Instructions.Count;
                Inst last_instruction = node.Instructions[node_instruction_count - 1];

                Mono.Cecil.Cil.OpCode op = last_instruction.OpCode;
                Mono.Cecil.Cil.FlowControl fc = op.FlowControl;

                // Add jump edge.
                switch (fc)
                {
                    case Mono.Cecil.Cil.FlowControl.Branch:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                        {
                            // Two cases: i.Operand is a single instruction, or an array of instructions.
                            if (last_instruction.Operand as Mono.Cecil.Cil.Instruction != null)
                            {
                                Mono.Cecil.Cil.Instruction target_instruction =
                                    last_instruction.Operand as Mono.Cecil.Cil.Instruction;
                                CFG.Vertex target_node = _cfg.VertexNodes.First(
                                    (CFG.Vertex x) =>
                                    {
                                        if (!x.Instructions.First().Instruction.Equals(target_instruction))
                                            return false;
                                        return true;
                                    });
                                _cfg.AddEdge(node, target_node);
                            }
                            else if (last_instruction.Operand as Mono.Cecil.Cil.Instruction[] != null)
                            {
                                foreach (Mono.Cecil.Cil.Instruction target_instruction in
                                    (last_instruction.Operand as Mono.Cecil.Cil.Instruction[]))
                                {
                                    CFG.Vertex target_node = _cfg.VertexNodes.First(
                                        (CFG.Vertex x) =>
                                        {
                                            if (!x.Instructions.First().Instruction.Equals(target_instruction))
                                                return false;
                                            return true;
                                        });
                                    _cfg.AddEdge(node, target_node);
                                }
                            }
                            else
                                throw new Exception("Unknown operand type for conditional branch.");
                            break;
                        }
                }

                // Add fall through edge.
                switch (fc)
                {
                    //case Mono.Cecil.Cil.FlowControl.Branch:
                    //case Mono.Cecil.Cil.FlowControl.Break:
                    case Mono.Cecil.Cil.FlowControl.Call:
                    case Mono.Cecil.Cil.FlowControl.Cond_Branch:
                    case Mono.Cecil.Cil.FlowControl.Meta:
                    case Mono.Cecil.Cil.FlowControl.Next:
                    case Mono.Cecil.Cil.FlowControl.Phi:
                    //case Mono.Cecil.Cil.FlowControl.Return:
                    //case Mono.Cecil.Cil.FlowControl.Throw:
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
                            var owner = _cfg.VertexNodes.Where(
                                n => n.Instructions.Where(ins => ins.Instruction == next_instruction).Any()).ToList();
                            if (owner.Count != 1)
                                throw new Exception("Cannot find instruction!");
                            CFG.Vertex target_node = owner.FirstOrDefault();
                            _cfg.AddEdge(node, target_node);
                        }
                        break;
                }
            }
            _cfg.OutputDotGraph();
            _cfg.OutputEntireGraph();
            System.Console.WriteLine();

            //this.Dump();

            // At this point, the given method has been converted into a bunch of basic blocks,
            // and are part of the CFG.
        }

        public MethodDefinition Rewrite(MethodReference method_reference)
        {
            // Look up base class.
            TypeReference mr_dt = method_reference.DeclaringType;
            if (mr_dt != null && mr_dt.FullName == "System.String")
            {
                // Find in Campy.Runtime.
                string yopath = @"C:\Users\Kenne\Documents\DotNetAnywhere\corlib\bin\Debug\corlib.dll";
                Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(yopath);
                foreach (var ttt in md.Types)
                {
                    var yyy = ttt.Resolve();
                    foreach (var mmm in yyy.Methods)
                    {
                        System.Console.WriteLine("type " + yyy.FullName + " Method " + mmm.FullName);
                        var bbb = mmm.Body;
                    }
                }
            }

            MethodDefinition method_definition = method_reference.Resolve();
            var name = method_reference.FullName;
            var found = _rewritten_runtime.ContainsKey(name);
            if (found)
            {
                string rewrite = _rewritten_runtime[name];
                MethodDefinition def = GetDefinition(rewrite);
                method_definition = def;
            }
            if (method_definition == null)
            {
                // Note, some situations MethodDefinition.Resolve() return null.
                // For a good introduction on Resolve(), see https://github.com/jbevain/cecil/wiki/Resolving
                // According to that, it should always return non-null. However, multi-dimensional arrays do not
                // seem to resolve. Moreover, to add more confusion, a one dimensional array do resolve,
                // As arrays are so central to GPU programming, we need to substitute for System.Array code--that
                // we cannot find--into code from a runtime library.
                System.Console.WriteLine("ERROR: No definition for " + method_reference.FullName);
                System.Console.WriteLine(method_reference.IsDefinition ? "" : "Is not a definition at that!");
                this.Failed = true;
                return null;
            }
            else if (method_definition.Body == null)
            {
                System.Console.WriteLine("ERROR: METHOD BODY NULL! " + method_definition);
                var t = method_definition.IsInternalCall;
                this.Failed = true;
                return null;
            }

            Dictionary<TypeReference, System.Type> additional = new Dictionary<TypeReference, System.Type>();
            var mr_gp = method_reference.GenericParameters;
            var mr_hgp = method_reference.HasGenericParameters;
            var mr_dt_hgp = method_reference.DeclaringType.HasGenericParameters;
            var mr_igi = method_reference.IsGenericInstance;
            var mr_dt_igi = method_reference.DeclaringType.IsGenericInstance;
            if (mr_igi)
            {
                GenericInstanceMethod i = method_reference as GenericInstanceMethod;
                var mr_hga = i.HasGenericArguments;
            }
            if (mr_dt_igi)
            {
                GenericInstanceType git = mr_dt as GenericInstanceType;
                var mr_dt_hga = git.HasGenericArguments;
                Collection<TypeReference> ga = git.GenericArguments;
                var e1 = git.ElementType;
                var e2 = git.GetElementType();
                Collection<GenericParameter> gg = e1.GenericParameters;
                // Map parameter to instantiated type.
                for (int i = 0; i < gg.Count; ++i)
                {
                    GenericParameter pp = gg[i];
                    TypeReference qq = ga[i];
                    TypeReference trrr = pp as TypeReference;
                    var system_type = qq
                        .ToSystemType();
                    if (system_type == null) throw new Exception("Failed to convert " + qq);
                    additional[pp] = system_type;
                }
            }
            return method_definition;
        }

        private void RewriteCilCodeBlock(Mono.Cecil.Cil.MethodBody body)
        {
            List<Instruction> result = new List<Instruction>();
            for (int j = 0; j < body.Instructions.Count; ++j)
            {
                Instruction i = body.Instructions[j];

                var inst_to_insert = i;

                if (i.OpCode.FlowControl == FlowControl.Call)
                {
                    object method = i.Operand;

                    if (method as Mono.Cecil.MethodReference == null)
                        throw new Exception();

                    Mono.Cecil.MethodReference method_reference = method as Mono.Cecil.MethodReference;

                    var name = method_reference.FullName;
                    var found = _rewritten_runtime.ContainsKey(name);
                    if (found)
                    {
                        string rewrite = _rewritten_runtime[name];
                        MethodDefinition def = GetDefinition(rewrite);

                        CallSite cs = new CallSite(typeof(void).ToMonoTypeReference());
                        body.Instructions.RemoveAt(j);
                        var worker = body.GetILProcessor();
                        Instruction new_inst = worker.Create(i.OpCode, def);
                        body.Instructions.Insert(j, new_inst);
                    }
                }
            }
        }
    }
}
