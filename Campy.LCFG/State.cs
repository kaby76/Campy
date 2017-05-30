using System;
using System.Collections.Generic;
using System.Diagnostics;
using Campy.CIL;
using Campy.Graphs;
using Campy.Utils;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public class State
    {
        // See ECMA 335, page 82.
        public StackQueue<Value> _stack;
        public ArraySection<Value> _arguments;
        public ArraySection<Value> _locals;
        public Dictionary<String, Value> _memory;

        public State()
        {
            _arguments = null;
            _locals = null;
            _memory = new Dictionary<string, Value>();
        }

        public State(CIL_CFG.Vertex node, LLVMCFG.Vertex llvm_node, int level)
        {
            int args = 0;
            args = node.NumberOfArguments;
            Debug.Assert(args == node.Method.Parameters.Count + (node.Method.HasThis ? 1 : 0));
            int locals = 0;
            locals = node.NumberOfLocals;
            Debug.Assert(locals == node.Method.Body.Variables.Count);
            _stack = new StackQueue<Value>();

            // State depends on predecessors. To handle this without updating state
            // until a fix point is found while converting to LLVM IR, we introduce
            // SSA phi functions.
            if (node.IsEntry)
            {
                var fun = llvm_node.Function;
                _arguments = _stack.Section(_stack.Count, args);
                for (uint i = 0; i < args; ++i)
                {
                    var par = LLVM.GetParam(fun, i);
                    var vx = new Value(par);
                    _stack.Push(vx);
                }
                _locals = _stack.Section(_stack.Count, locals);
                for (int i = 0; i < locals; ++i)
                    _stack.Push(default(Value));
                for (int i = _stack.Size(); i < level; ++i)
                    _stack.Push(default(Value));
            }
            else if (node._Predecessors.Count == 1)
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
            else if (node._Predecessors.Count == 0)
            {
                // This is a problem because it's a dead block. We shouldn't
                // even be processing this bb.
                // TODO
            }
            else // node._Predecessors.Count > 0
            {
                // Now, for every arg, local, stack, set up for merge.
                var pred = llvm_node._Predecessors[0].From;
                var p_llvm_node = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(pred)];
                var size = p_llvm_node.StateOut._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    _stack.Push(default(Value));
                    var count = node._Predecessors.Count;
                    ValueRef res = LLVM.BuildPhi(llvm_node.Builder, LLVM.Int32Type(), "");
                    ValueRef[] phi_vals = new ValueRef[count];
                    for (int c = 0; c < count; ++c)
                    {
                        var p = llvm_node._Predecessors[c].From;
                        var plm = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(p)];
                        var vr = plm.StateOut._stack[i];
                        phi_vals[c] = vr.V;
                    }
                    BasicBlockRef[] phi_blocks = new BasicBlockRef[count];
                    for (int c = 0; c < count; ++c)
                    {
                        var p = llvm_node._Predecessors[c].From;
                        var plm = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(p)];
                        phi_blocks[c] = plm.BasicBlock;
                    }
                    LLVM.AddIncoming(res, phi_vals, phi_blocks);
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
            System.Console.Write("[args");
            for (int i = 0; i < args; ++i)
                System.Console.Write(" " + _stack[i]);
            System.Console.Write("]");
            System.Console.Write("[locs");
            for (int i = 0; i < locs; ++i)
                System.Console.Write(" " + _stack[args + i]);
            System.Console.Write("]");
            for (int i = args + locs; i < _stack.Size(); ++i)
                System.Console.Write(" " + _stack[i]);
            System.Console.WriteLine();
        }
    }
}
