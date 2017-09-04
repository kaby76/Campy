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
        public ArraySection<Value> _this; // Pointer to _stack, if there is a "this" pointer.
        public ArraySection<Value> _arguments; // Pointer to _stack, if there are parameters for the method.
        public ArraySection<Value> _locals; // Pointer to _stack, if there are local variables to the method.
        public Dictionary<String, Value> _memory;
        public List<ValueRef> _phi;

        public State()
        {
            _stack = new StackQueue<Value>();
            _this = null;
            _arguments = null;
            _locals = null;
            _memory = new Dictionary<string, Value>();
            _phi = new List<ValueRef>();
        }

        public State(CFG.Vertex bb, MethodDefinition md, int args, int locals, int level)
        {
            // Set up state with args, locals, basic stack initial value of 0xDEADBEEF.
            // In addition, use type information from method to compute types for all args.
            _stack = new StackQueue<Value>();

            int begin = 0;

            if (md.HasThis)
            {
                TypeDefinition td = md.DeclaringType;
                TypeRef type = Converter.ConvertMonoTypeToLLVM(bb, td, bb.LLVMTypeMap, bb.OpsFromOriginal);
                var vx = new Value(LLVM.ConstInt(type, (ulong)0xdeadbeef, true));
                _stack.Push(vx);
                _this = _stack.Section(begin++, 1);
            }
            else
            {
                _this = _stack.Section(begin, 0);
            }

            for (int i = begin; i < level; ++i)
            {
                TypeRef type = LLVM.Int32Type();
                if (i < begin + args)
                {
                    int j = i - begin;
                    ParameterDefinition p = md.Parameters[j];
                    TypeReference tr = p.ParameterType;
                    type = Converter.ConvertMonoTypeToLLVM(bb, tr, bb.LLVMTypeMap, bb.OpsFromOriginal);
                }
                var vx = new Value(LLVM.ConstInt(type, (ulong)0xdeadbeef, true));
                _stack.Push(vx);
            }
            _arguments = _stack.Section(0 /* NB: args begin with "this" ptr. */, args + begin);
            _locals = _stack.Section(args + begin, locals);
            _phi = new List<ValueRef>();
        }

        public State(Dictionary<int, bool> visited, CFG.Vertex bb, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            // Set up a blank stack.
            _stack = new StackQueue<Value>();

            int args = bb.NumberOfArguments;
            int locals = bb.NumberOfLocals;
            int level = (int)bb.StackLevelIn;

            // Set up list of phi functions in case there are multiple predecessors.
            _phi = new List<ValueRef>();

            // State depends on predecessors. To handle this without updating state
            // until a fix point is found while converting to LLVM IR, we introduce
            // SSA phi functions.
            if (bb._Predecessors.Count == 0)
            {
                if (!bb.IsEntry) throw new Exception("Cannot handle dead code blocks.");
                var fun = bb.Function;
                uint begin = 0;
                if (bb.HasThis)
                {
                    var par = LLVM.GetParam(fun, begin++);
                    var vx = new Value(par);
                    System.Console.WriteLine("in state() " + vx.ToString());
                    _stack.Push(vx);
                    _this = _stack.Section((int)0, 1);
                }
                else
                {
                    _this = _stack.Section((int)0, 0);
                }
                // Set up args. NB: Args begin with "this" pointer according to spec!!!!!
                _arguments = _stack.Section((int)0, args + (int)begin);
                for (uint i = begin; i < args + begin; ++i)
                {
                    ValueRef par = LLVM.GetParam(fun, i);
                    var vx = new Value(par);
                    System.Console.WriteLine(" " + vx);
                    _stack.Push(vx);
                }

                // Set up locals. I'm making an assumption that there is a 
                // one to one and in order mapping of the locals with that
                // defined for the method body by Mono.
                Collection<VariableDefinition> variables = bb.Method.Body.Variables;
                _locals = _stack.Section((int)(args+begin), locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i].VariableType;
                    TypeRef type = Converter.ConvertMonoTypeToLLVM(bb, tr, bb.LLVMTypeMap, bb.OpsFromOriginal);
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
            else if (bb._Predecessors.Count == 1)
            {
                // We don't need phi functions--and can't with LLVM--
                // if there is only one predecessor. If it hasn't been
                // converted before this node, just create basic state.

                var pred = bb._Predecessors[0].From;
                var p_llvm_node = bb._Graph.VertexSpace[bb._Graph.NameSpace.BijectFromBasetype(pred)];
                var other = p_llvm_node.StateOut;
                var size = p_llvm_node.StateOut._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    var vx = other._stack[i];
                    _stack.Push(vx);
                }
                _this = _stack.Section(other._this.Base, other._this.Len);
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
                pred = bb._Predecessors[0].From;
                for (int pred_ind = 0; pred_ind < bb._Predecessors.Count; ++pred_ind)
                {
                    int to_check = bb._Predecessors[pred_ind].From;
                    if (!visited.ContainsKey(to_check)) continue;
                    CFG.Vertex check_llvm_node = bb._Graph.VertexSpace[bb._Graph.NameSpace.BijectFromBasetype(to_check)];
                    if (check_llvm_node.StateOut == null)
                        continue;
                    if (check_llvm_node.StateOut._stack == null)
                        continue;
                    pred = pred_ind;
                    break;
                }

                CFG.Vertex p_llvm_node = bb._Graph.VertexSpace[bb._Graph.NameSpace.BijectFromBasetype(bb._Predecessors[pred].From)];
                int size = p_llvm_node.StateOut._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    {
                        Value value = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
                        _stack.Push(value);
                    }
                    var count = bb._Predecessors.Count;
                    var v = p_llvm_node.StateOut._stack[i].V;
                    TypeRef tr = LLVM.TypeOf(v);
                    ValueRef res = LLVM.BuildPhi(bb.Builder, tr, "");
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
                _this = _stack.Section(other._this.Base, other._this.Len);
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
            _this = _stack.Section(other._this.Base, other._this.Len);
            _arguments = _stack.Section(other._arguments.Base, other._arguments.Len);
            _locals = _stack.Section(other._locals.Base, other._locals.Len);
        }

        public void Dump()
        {
            //return;
            int args = _arguments.Len;
            int locs = _locals.Len;
            int begin = 0;
            System.Console.WriteLine("This size = " + _this.Len);
            System.Console.WriteLine("Args size = " + _arguments.Len);
            System.Console.WriteLine("Locals size = " + _locals.Len);
            System.Console.WriteLine("Stack size = " + _stack.Count);
            if (_this.Len > 0)
            {
                System.Console.WriteLine("[this (base " + _this.Base + ")");
                System.Console.WriteLine(_this[0]);
                System.Console.WriteLine();
                System.Console.WriteLine("]");
                System.Console.WriteLine();
            }
            System.Console.WriteLine("[args (base " + _arguments.Base + ")");
            for (int i = 0; i < args; ++i)
            {
                System.Console.WriteLine(" " + _arguments[i]);
                System.Console.WriteLine();
            }
            System.Console.WriteLine("]");
            System.Console.WriteLine();
            System.Console.WriteLine("[locs (base " + _locals.Base + ")");
            for (int i = 0; i < locs; ++i)
            {
                System.Console.WriteLine(" " + _locals[i]);
                System.Console.WriteLine();
            }
            System.Console.WriteLine("]");
            System.Console.WriteLine();
            System.Console.WriteLine("[rest of stack (base " + (args + locs) + ")");
            // NB. Args includes "this" pointer.
            for (int i = args + locs; i < _stack.Size(); ++i)
            {
                System.Console.WriteLine(" " + _stack[i]);
                System.Console.WriteLine();
            }
            System.Console.WriteLine("]");
            System.Console.WriteLine();
            System.Console.WriteLine("[complete stack (base " + 0 + ")");
            for (int i = 0; i < _stack.Size(); ++i)
            {
                System.Console.WriteLine(" " + _stack[i]);
                System.Console.WriteLine();
            }
            System.Console.WriteLine("]");
            System.Console.WriteLine();
        }
    }
}
