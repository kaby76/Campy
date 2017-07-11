using System;
using System.Collections.Generic;
using System.Diagnostics;
using Campy.Graphs;
using Campy.Utils;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Collections.Generic;
using Swigged.LLVM;

namespace Campy.ControlFlowGraph
{
    public class State
    {
        // See ECMA 335, page 82.
        public StackQueue<Value> _stack;
        public ArraySection<Value> _arguments;
        public ArraySection<Value> _locals;
        public Dictionary<String, Value> _memory;
        public List<ValueRef> _phi;

        public State()
        {
            _arguments = null;
            _locals = null;
            _memory = new Dictionary<string, Value>();
            _phi = new List<ValueRef>();
        }

        public State(MethodDefinition md, int args, int locals, int level)
        {
            // Set up state with args, locals, basic stack initial value of 0xDEADBEEF.
            // In addition, use type information from method to compute types for all args.
            _stack = new StackQueue<Value>();
            for (int i = 0; i < level; ++i)
            {
                TypeRef type = LLVM.Int32Type();
                if (i < args)
                {
                    if (md.HasThis && i == 0)
                    {
                        // First parameter is "this", the object that method is attached to.
                        // We'll record a pointer to the object type.
                        var td = md.DeclaringType;
                        System.Type sys_td = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(td);
                        type = Converter.ConvertSystemTypeToLLVM(sys_td, false);
                    }
                    else
                    {
                        int j = md.HasThis ? i - 1 : i;
                        ParameterDefinition p = md.Parameters[j];
                        TypeReference tr = p.ParameterType;
                        TypeDefinition td = tr.Resolve();
                        System.Type sys_td = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(tr);
                        type = Converter.ConvertSystemTypeToLLVM(sys_td, false);
                    }
                }
                var vx = new Value(LLVM.ConstInt(type, (ulong)0xdeadbeef, true));
                _stack.Push(vx);
            }
            _arguments = _stack.Section(0, args);
            _locals = _stack.Section(args, locals);
            _phi = new List<ValueRef>();
        }

        public State(Dictionary<int, bool> visited, CFG.Vertex llvm_node)
        {
            int args = llvm_node.NumberOfArguments;
            int locals = llvm_node.NumberOfLocals;
            int level = (int)llvm_node.StackLevelIn;

            // Set up list of phi functions in case there are multiple predecessors.
            _phi = new List<ValueRef>();

            // Set up a blank stack.
            _stack = new StackQueue<Value>();

            // State depends on predecessors. To handle this without updating state
            // until a fix point is found while converting to LLVM IR, we introduce
            // SSA phi functions.
            if (llvm_node._Predecessors.Count == 0)
            {
                if (!llvm_node.IsEntry) throw new Exception("Cannot handle dead code blocks.");
                var fun = llvm_node.Function;

                // Set up args.
                _arguments = _stack.Section(0, args);
                for (uint i = 0; i < args; ++i)
                {
                    var par = LLVM.GetParam(fun, i);
                    var vx = new Value(par);
                    _stack.Push(vx);
                }

                // Set up locals. I'm making an assumption that there is a 
                // one to one and in order mapping of the locals with that
                // defined for the method body by Mono.
                Collection<VariableDefinition> variables = llvm_node.Method.Body.Variables;
                _locals = _stack.Section(args, locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i].VariableType;
                    var td = tr.Resolve();
                    System.Type sys_td = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(tr);
                    TypeRef type = Converter.ConvertSystemTypeToLLVM(sys_td, false);
                    Value value = new Value(LLVM.ConstInt(type, (ulong)0, true));
                    _stack.Push(value);
                }

                // Set up any thing else.
                for (int i = _stack.Size(); i < level; ++i)
                {
                    Value value = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
                    _stack.Push(value);
                }
            }
            else if (llvm_node._Predecessors.Count == 1)
            {
                // We don't need phi functions--and can't with LLVM--
                // if there is only one predecessor. If it hasn't been
                // converted before this node, just create basic state.

                var pred = llvm_node._Predecessors[0].From;
                var p_llvm_node = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(pred)];
                var other = p_llvm_node.StateOut;
                var size = p_llvm_node.StateOut._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    var vx = other._stack[i];
                    _stack.Push(vx);
                }
                _arguments = _stack.Section(other._arguments.Base, other._arguments.Len);
                _locals = _stack.Section(other._locals.Base, other._locals.Len);
            }
            else // node._Predecessors.Count > 0
            {
                // As we cannot guarentee whether all predecessors are fulfilled,
                // make up something so we don't have problems.
                // Now, for every arg, local, stack, set up for merge.
                // Find a predecessor that has some definition.
                int pred = -1;
                pred = llvm_node._Predecessors[0].From;
                for (int pred_ind = 0; pred_ind < llvm_node._Predecessors.Count; ++pred_ind)
                {
                    int to_check = llvm_node._Predecessors[pred_ind].From;
                    if (!visited.ContainsKey(to_check)) continue;
                    CFG.Vertex check_llvm_node = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(to_check)];
                    if (check_llvm_node.StateOut == null)
                        continue;
                    if (check_llvm_node.StateOut._stack == null)
                        continue;
                    pred = pred_ind;
                    break;
                }

                CFG.Vertex p_llvm_node = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(llvm_node._Predecessors[pred].From)];
                int size = p_llvm_node.StateOut._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    {
                        Value value = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
                        _stack.Push(value);
                    }
                    var count = llvm_node._Predecessors.Count;
                    var v = p_llvm_node.StateOut._stack[i].V;
                    TypeRef tr = LLVM.TypeOf(v);
                    ValueRef res = LLVM.BuildPhi(llvm_node.Builder, tr, "");
                    _phi.Add(res);
                    
                    //ValueRef[] phi_vals = new ValueRef[count];
                    //for (int c = 0; c < count; ++c)
                    //{
                    //    var p = llvm_node._Predecessors[c].From;
                    //    var plm = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(p)];
                    //    var vr = plm.StateOut._stack[i];
                    //    phi_vals[c] = vr.V;
                    //}
                    //BasicBlockRef[] phi_blocks = new BasicBlockRef[count];
                    //for (int c = 0; c < count; ++c)
                    //{
                    //    var p = llvm_node._Predecessors[c].From;
                    //    var plm = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(p)];
                    //    phi_blocks[c] = plm.BasicBlock;
                    //}
                    //LLVM.AddIncoming(res, phi_vals, phi_blocks);
                    _stack[i] = new Value(res);
                }
                var other = p_llvm_node.StateOut;
                _arguments = _stack.Section(other._arguments.Base, other._arguments.Len);
                _locals = _stack.Section(other._locals.Base, other._locals.Len);
            }
        }

        public State(State other)
        {
            _stack = new StackQueue<Value>();
            for (int i = 0; i < other._stack.Count; ++i)
            {
                _stack.Push(other._stack.PeekBottom(i));
            }
            _arguments = _stack.Section(other._arguments.Base, other._arguments.Len);
            _locals = _stack.Section(other._locals.Base, other._locals.Len);
        }

        public void Dump()
        {
            int args = _arguments.Len;
            int locs = _locals.Len;
            System.Console.WriteLine("[args");
            for (int i = 0; i < args; ++i)
            {
                System.Console.WriteLine(" " + _stack[i]);
            }
            System.Console.WriteLine("]");
            System.Console.WriteLine("[locs");
            for (int i = 0; i < locs; ++i)
                System.Console.WriteLine(" " + _stack[args + i]);
            System.Console.WriteLine("]");
            for (int i = args + locs; i < _stack.Size(); ++i)
                System.Console.WriteLine(" " + _stack[i]);
            System.Console.WriteLine();
        }
    }
}
