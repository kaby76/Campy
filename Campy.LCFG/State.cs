using System;
using System.Collections.Generic;
using System.Diagnostics;
using Campy.CIL;
using Campy.Utils;
using Swigged.LLVM;

namespace Campy.LCFG
{
    // Call stack handled by pointers to previous
    // call frame, and additional assignments for
    // args/params. When evaluating, the call stack
    // of the current instruction must match call
    // frame for which it is being evaluated.
    public class Nesting
    {
        public Inst _caller;
        public List<Inst> _parameter_argument_matching = new List<Inst>();
        public Nesting _previous;
    }

    public class State
    {
        public List<Nesting> _bindings;

        // See ECMA 335, page 82.
        public StackQueue<Value> _stack;

        // _arguments and _locals are objects that actually point
        // to _stack.
        public ArraySection<Value> _arguments;
        public ArraySection<Value> _locals;

        // _memory contains all objects that are allocated
        // such as classes and arrays.
        public Dictionary<String, Value> _memory;

        public State()
        {
            _arguments = null;
            _locals = null;
            _memory = new Dictionary<string, Value>();
            _bindings = new List<Nesting>();
            _bindings.Add(new Nesting());
        }

        public State(CIL_CFG.Vertex node, LLVMCFG.Vertex llvm_node, int level)
        {
            int args = 0;
            //if (md.HasThis) args++;
            //args += md.Parameters.Count;
            args = node.NumberOfArguments;
            Debug.Assert(args == node.Method.Parameters.Count + (node.Method.HasThis ? 1 : 0));
            int locals = 0;
            //locals = md.Body.Variables.Count;
            locals = node.NumberOfLocals;
            Debug.Assert(locals == node.Method.Body.Variables.Count);

            // Create a stack with variables, which will be
            // bound to phi functions.
            _stack = new StackQueue<Value>();

            // Allocate parameters.
            _arguments = _stack.Section(_stack.Count, args);
            for (uint i = 0; i < args; ++i)
            {
                var fun = llvm_node.Function;
                var par = LLVM.GetParam(fun, i);
                var vx = new Value(par);
                //var phi = new Phi();
                //phi_functions[vx] = phi;
                List<Value> list = new List<Value>();
                //phi._merge = list;
                //phi._v = vx;
                _stack.Push(vx);
            }

            // Allocate local variables.
            _locals = _stack.Section(_stack.Count, locals);
            for (int i = 0; i < locals; ++i)
                _stack.Push(default(Value));

            for (int i = _stack.Size(); i < level; ++i)
                _stack.Push(default(Value));

            _bindings = new List<Nesting>();
            _bindings.Add(new Nesting());
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
            _bindings = new List<Nesting>();
            _bindings.AddRange(other._bindings);
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
