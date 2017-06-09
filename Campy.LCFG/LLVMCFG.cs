using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Campy.Graphs;
using Campy.Utils;
using Mono.Cecil;
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

            public List<Inst> Instructions { get; } = new List<Inst>();

            public int NumberOfArguments { get; set; }
            public int NumberOfLocals { get; set; }
            public bool IsEntry { get; set; }
            public MethodDefinition Method { get; set; }
            public bool HasReturnValue { get; set; }

            public int? StackLevelIn { get; set; }

            public int? StackLevelOut { get; set; }

            public State StateIn { get; set; }

            public State StateOut { get; set; }

            public Vertex()
                : base()
            {
            }

            public Vertex(BasicBlockRef bb)
            {
                BasicBlock = bb;
            }
        }

        public class Edge
            : GraphLinkedList<int, Vertex, Edge>.Edge
        { }

    }
}
