using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Campy.Graphs;
using Campy.Utils;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public class LLVMCFG : GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>
    {
        public class Vertex
            : GraphLinkedList<int, Vertex, Edge>.Vertex
        {
            public BasicBlockRef BasicBlock { get; set; }
            public ValueRef Function { get; set; }
            public BuilderRef Builder { get; set; }
            public ModuleRef Module { get; set; }

            private List<Inst> _instructions = new List<Inst>();
            public List<Inst> Instructions
            {
                get
                {
                    return _instructions;
                }
            }

            private State _state_in;
            public State StateIn
            {
                get { return _state_in; }
                set { _state_in = value; }
            }

            private State _state_out;
            public State StateOut
            {
                get { return _state_out; }
                set { _state_out = value; }
            }

            public Vertex()
                : base()
            {
                _state_in = new State();
                _state_out = new State();
            }

            public Vertex(BasicBlockRef bb)
            {
                BasicBlock = bb;
                _state_in = new State();
                _state_out = new State();
            }
        }

        public class Edge
            : GraphLinkedList<int, Vertex, Edge>.Edge
        { }

    }
}
