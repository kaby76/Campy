﻿using System;
using System.Collections.Generic;
using Campy.Utils;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Collections.Generic;
using Swigged.LLVM;

namespace Campy.Compiler
{
    public class State
    {
        // See ECMA 335, page 82.
        public StackQueue<Value> _stack;
        public ListSection<Value> _struct_ret; // Pointer to _stack, if there is a "this" pointer.
        public ListSection<Value> _this; // Pointer to _stack, if there is a "this" pointer.
        public ListSection<Value> _arguments; // Pointer to _stack, if there are parameters for the method.
        public ListSection<Value> _locals; // Pointer to _stack, if there are local variables to the method.
        public Dictionary<String, Value> _memory;
        public List<ValueRef> _phi;

        public State()
        {
            _stack = new StackQueue<Value>();
            _this = null;
            _arguments = null;
            _locals = null;
            _struct_ret = null;
            _memory = new Dictionary<string, Value>();
            _phi = new List<ValueRef>();
        }

        public State(CFG.Vertex basic_block, bool use_in = true)
        {
            int level = use_in ? (int)basic_block.StackLevelIn : (int)basic_block.StackLevelOut;
            int args = basic_block.NumberOfArguments;
            bool scalar_ret = basic_block.HasScalarReturnValue;
            bool struct_ret = basic_block.HasStructReturnValue;
            bool has_this = basic_block.HasThis;
            int locals = basic_block.NumberOfLocals;

            // Set up state with args, locals, basic stack initial value of 0xDEADBEEF.
            // In addition, use type information from method to compute types for all args.
            _stack = new StackQueue<Value>();

            int begin = 0;

            _arguments = _stack.Section(0 /* NB: args begin with "this" ptr. */, args + begin);
            _locals = _stack.Section(args + begin, locals);
            _phi = new List<ValueRef>();

            var fun = basic_block.MethodValueRef;
            var t_fun = LLVM.TypeOf(fun);
            var t_fun_con = LLVM.GetTypeContext(t_fun);
            var context = LLVM.GetModuleContext(Converter.global_llvm_module);
            if (t_fun_con != context) throw new Exception("not equal");

            for (uint i = 0; i < args; ++i)
            {
                var par = new Value(LLVM.GetParam(fun, i));
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(par);
                _stack.Push(par);
            }

            if (scalar_ret)
                ; // No parameter to use for return value, just return directly in LLVM.

            int offset = 0;
            _struct_ret = _stack.Section(struct_ret ? offset++ : offset, struct_ret ? 1 : 0);
            _this = _stack.Section(has_this ? offset++ : offset, has_this ? 1 : 0);
            // Set up args. NB: arg 0 is "this" pointer according to spec!!!!!
            _arguments = _stack.Section(
                has_this ? offset - 1 : offset,
                args - (struct_ret ? 1 : 0));

            // Set up locals. I'm making an assumption that there is a 
            // one to one and in order mapping of the locals with that
            // defined for the method body by Mono.
            Collection<VariableDefinition> variables = basic_block.RewrittenCalleeSignature.Resolve().Body.Variables;
            _locals = _stack.Section((int)_stack.Count, locals);
            for (int i = 0; i < locals; ++i)
            {
                var tr = variables[i].VariableType;
                Type type = new Type(tr);
                Value value;
                if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.PointerTypeKind)
                    value = new Value(LLVM.ConstPointerNull(type.IntermediateType));
                else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.DoubleTypeKind)
                    value = new Value(LLVM.ConstReal(LLVM.DoubleType(), 0));
                else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.IntegerTypeKind)
                    value = new Value(LLVM.ConstInt(type.IntermediateType, (ulong)0, true));
		        else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.StructTypeKind)
		        {
			        var entry = basic_block.Entry.BasicBlock;
			        var beginning = LLVM.GetFirstInstruction(entry);
			        LLVM.PositionBuilderBefore(basic_block.Builder, beginning);
			        var new_obj = LLVM.BuildAlloca(basic_block.Builder, type.IntermediateType, ""); // Allocates struct on stack, but returns a pointer to struct.
		            var stuff = LLVM.BuildLoad(basic_block.Builder, new_obj, "");
			        LLVM.PositionBuilderAtEnd(basic_block.Builder, basic_block.BasicBlock);
			        value = new Value(stuff);
		        }
                else
                    throw new Exception("Unhandled type");
                _stack.Push(value);
            }

            // Set up any thing else.
            for (int i = _stack.Size(); i < level; ++i)
            {
                Value value = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
                _stack.Push(value);
            }
        }

        public State(Dictionary<int, bool> visited, CFG.Vertex bb, List<Mono.Cecil.TypeReference> list_of_data_types_used)
        {
            // Set up a blank stack.
            _stack = new StackQueue<Value>();

            int args = bb.NumberOfArguments;
            bool scalar_ret = bb.HasScalarReturnValue;
            bool struct_ret = bb.HasStructReturnValue;
            bool has_this = bb.HasThis;
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
                var fun = bb.MethodValueRef;
                var t_fun = LLVM.TypeOf(fun);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(Converter.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");

                for (uint i = 0; i < args; ++i)
                {
                    var par = new Value(LLVM.GetParam(fun, i));
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(par);
                    _stack.Push(par);
                }

                if (scalar_ret)
                    ; // No parameter to use for return value, just return directly in LLVM.

                int offset = 0;
                _struct_ret = _stack.Section(struct_ret ? offset++ : offset, struct_ret ? 1 : 0);
                _this = _stack.Section(has_this ? offset++ : offset, has_this ? 1 : 0);
                // Set up args. NB: arg 0 is "this" pointer according to spec!!!!!
                _arguments = _stack.Section(
                    has_this ? offset - 1 : offset,
                    args - (struct_ret ? 1 : 0));

                // Set up locals. I'm making an assumption that there is a 
                // one to one and in order mapping of the locals with that
                // defined for the method body by Mono.
                Collection<VariableDefinition> variables = bb.RewrittenCalleeSignature.Resolve().Body.Variables;
                _locals = _stack.Section((int)_stack.Count, locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i].VariableType;
                    Type type = new Type(tr);
                    Value value;
                    if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.PointerTypeKind)
                        value = new Value(LLVM.ConstPointerNull(type.IntermediateType));
                    else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.DoubleTypeKind)
                        value = new Value(LLVM.ConstReal(LLVM.DoubleType(), 0));
                    else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.IntegerTypeKind)
                        value = new Value(LLVM.ConstInt(type.IntermediateType, (ulong)0, true));
                    else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.StructTypeKind)
                        value = new Value(LLVM.ConstPointerNull(type.IntermediateType));
                    else
                        throw new Exception("Unhandled type");
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
                _struct_ret = _stack.Section(other._struct_ret.Base, other._struct_ret.Len);
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
                        Value f = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
                        _stack.Push(f);
                    }
                    var count = bb._Predecessors.Count;
                    var value = p_llvm_node.StateOut._stack[i];
                    var v = value.V;
                    TypeRef tr = LLVM.TypeOf(v);
                    ValueRef res = LLVM.BuildPhi(bb.Builder, tr, "");
                    _phi.Add(res);
                    _stack[i] = new Value(res);
                }
                var other = p_llvm_node.StateOut;
                _struct_ret = _stack.Section(other._struct_ret.Base, other._struct_ret.Len);
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
            _struct_ret = _stack.Section(other._struct_ret.Base, other._struct_ret.Len);
            _this = _stack.Section(other._this.Base, other._this.Len);
            _arguments = _stack.Section(other._arguments.Base, other._arguments.Len);
            _locals = _stack.Section(other._locals.Base, other._locals.Len);
        }

        public void OutputTrace()
        {
            int args = _arguments.Len;
            int locs = _locals.Len;
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
