using System;

namespace Campy.Compiler.Graph
{
    public interface IInst
    {
        Mono.Cecil.Cil.Instruction Instruction { get; set; }
        void Replace(Mono.Cecil.Cil.Instruction inst);
    }
}
