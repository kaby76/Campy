using System;
using System.Collections.Generic;
using Campy.Utils;

namespace Campy.Compiler
{
    public class STATE<T>
    {
        // See ECMA 335, page 82.
        public StackQueue<T> _stack;
        public ListSection<T> _struct_ret; // Pointer to _stack, if there is a "this" pointer.
        public ListSection<T> _this; // Pointer to _stack, if there is a "this" pointer.
        public ListSection<T> _arguments; // Pointer to _stack, if there are parameters for the method.
        public ListSection<T> _locals; // Pointer to _stack, if there are local variables to the method.

        public STATE()
        {
            _stack = new StackQueue<T>();
            _this = null;
            _arguments = null;
            _locals = null;
            _struct_ret = null;
        }

        public STATE(Dictionary<CFG.Vertex, bool> visited,
            Dictionary<CFG.Vertex, STATE<T>> states_in,
            Dictionary<CFG.Vertex, STATE<T>> states_out,
            CFG.Vertex bb,
            Action<STATE<T>, Dictionary<CFG.Vertex, STATE<T>>,Dictionary<CFG.Vertex, STATE<T>>,CFG.Vertex> custum_initializer
            )
        {
            // Set up a state that is a copy of another state.
            _stack = new StackQueue<T>();
            int in_level = -1;
            int args = bb.StackNumberOfArguments;
            bool scalar_ret = bb.HasScalarReturnValue;
            bool struct_ret = bb.HasStructReturnValue;
            bool has_this = bb.HasThis;
            int locals = bb.StackNumberOfLocals;
            // Use predecessor information to get initial stack size.
            if (bb.IsEntry)
            {
                in_level = bb.StackNumberOfLocals + bb.StackNumberOfArguments;
            }
            else
            {
                foreach (CFG.Vertex pred in bb._graph.PredecessorNodes(bb))
                {
                    // Do not consider interprocedural edges when computing stack size.
                    if (pred._original_method_reference != bb._original_method_reference)
                        throw new Exception("Interprocedural edge should not exist.");
                    // If predecessor has not been visited, warn and do not consider.
                    // Warn if predecessor does not concur with another predecessor.
                    if (in_level != -1 && states_out[pred]._stack.Count != in_level)
                        throw new Exception("Miscalculation in stack size "
                                            + "for basic block " + bb
                                            + " or predecessor " + pred);
                    in_level = states_out[pred]._stack.Count;
                }
            }
            if (in_level == -1)
            {
                throw new Exception("Predecessor edge computation screwed up.");
            }

            int level = in_level;

            custum_initializer(this, states_in, states_out, bb);
        }

        public STATE(STATE<T> other)
        {
            _stack = new StackQueue<T>();
            for (int i = 0; i < other._stack.Count; ++i)
            {
                _stack.Push(other._stack.PeekBottom(i));
            }
            _struct_ret = _stack.Section(other._struct_ret.Base, other._struct_ret.Len);
            _this = _stack.Section(other._this.Base, other._this.Len);
            _arguments = _stack.Section(other._arguments.Base, other._arguments.Len);
            _locals = _stack.Section(other._locals.Base, other._locals.Len);
        }

        public void OutputTrace(string indent)
        {
            int args = _arguments.Len;
            int locs = _locals.Len;
            System.Console.WriteLine(indent + "This size = " + _this.Len);
            System.Console.WriteLine(indent + "Args size = " + _arguments.Len);
            System.Console.WriteLine(indent + "Locals size = " + _locals.Len);
            System.Console.WriteLine(indent + "Stack size = " + _stack.Count);
            if (_this.Len > 0)
            {
                System.Console.WriteLine(indent + "[this (base " + _this.Base + ")");
                System.Console.WriteLine(indent + _this[0]);
                System.Console.WriteLine(indent + "]");
            }
            System.Console.WriteLine(indent + "[args (base " + _arguments.Base + ")");
            for (int i = 0; i < args; ++i)
            {
                System.Console.WriteLine(indent + _arguments[i]);
            }
            System.Console.WriteLine(indent + "]");
            System.Console.WriteLine(indent + "[locs (base " + _locals.Base + ")");
            for (int i = 0; i < locs; ++i)
            {
                System.Console.WriteLine(indent + _locals[i]);
            }
            System.Console.WriteLine(indent + "]");
            System.Console.WriteLine(indent + "[rest of stack (base " + (args + locs) + ")");
            // NB. Args includes "this" pointer.
            for (int i = args + locs; i < _stack.Size(); ++i)
            {
                System.Console.WriteLine(indent + _stack[i]);
            }
            System.Console.WriteLine(indent + "]");
            System.Console.WriteLine(indent + "[complete stack (base " + 0 + ")");
            for (int i = 0; i < _stack.Size(); ++i)
            {
                System.Console.WriteLine(indent + _stack[i]);
            }
            System.Console.WriteLine(indent + "]");
        }
    }
}
