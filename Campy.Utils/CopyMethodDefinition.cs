using System;
using System.Collections.Generic;
using System.Text;
using Mono.Cecil;
using Mono.Cecil.Cil;

namespace Campy.Utils
{
    public static class CopyMethodDefinition
    {
        // <summary>
        /// Copy a method from one module to another.  If the same method exists in the target module, the caller
        /// is responsible to delete it first.
        /// The sourceMethod makes calls to other methods, we divide the calls into two types:
        /// 1. MethodDefinition : these are methods that are defined in the same module as the sourceMethod;
        /// 2. MethodReference : these are methods that are defined in a different module
        /// For type 1 calls, we will copy these MethodDefinitions to the same target typedef.
        /// For type 2 calls, we will not copy the called method
        /// 
        /// Another limitation: any TypeDefinitions that are used in the sourceMethod will not be copied to the target module; a 
        /// typereference is created instead.
        /// </summary>
        /// <param name="copyToTypedef">The typedef to copy the method to</param>
        /// <param name="sourceMethod">The method to copy</param>
        /// <returns></returns>
        //public static MethodDefinition CopyMethod(MethodDefinition sourceMethod)
        //{

        //    ModuleDefinition targetModule = sourceMethod.DeclaringType.Module;

        //    // create a new MethodDefinition; all the content of sourceMethod will be copied to this new MethodDefinition

        //    MethodDefinition targetMethod = new MethodDefinition(sourceMethod.Name, sourceMethod.Attributes, targetModule.Import(sourceMethod.ReturnType));


        //    // Copy the parameters; 
        //    foreach (ParameterDefinition p in sourceMethod.Parameters)
        //    {
        //        ParameterDefinition nP = new ParameterDefinition(p.Name, p.Attributes, targetModule.Import(p.ParameterType));
        //        targetMethod.Parameters.Add(nP);
        //    }

        //    // copy the body
        //    MethodBody nBody = targetMethod.Body;
        //    MethodBody oldBody = sourceMethod.Body;

        //    nBody.InitLocals = oldBody.InitLocals;

        //    // copy the local variable definition
        //    foreach (VariableDefinition v in oldBody.Variables)
        //    {
        //        VariableDefinition nv = new VariableDefinition(v.Name, targetModule.Import(v.VariableType));
        //        nBody.Variables.Add(nv);
        //    }

        //    // copy the IL; we only need to take care of reference and method definitions
        //    Mono.Collections.Generic.Collection<Instruction> col = nBody.Instructions;
        //    foreach (Instruction i in oldBody.Instructions)
        //    {
        //        object operand = i.Operand;
        //        if (operand == null)
        //        {
        //            col.Add(new Instruction(i.OpCode, null));
        //            continue;
        //        }

        //        // for any methodef that this method calls, we will copy it

        //        if (operand is MethodDefinition)
        //        {
        //            MethodDefinition dmethod = operand as MethodDefinition;
        //            MethodDefinition newMethod = CopyMethod(copyToTypedef, dmethod);
        //            col.Add(new Instruction(i.OpCode, newMethod));
        //            continue;
        //        }

        //        // for member reference, import it
        //        if (operand is FieldReference)
        //        {
        //            FieldReference fref = operand as FieldReference;
        //            FieldReference newf = targetModule.Import(fref);
        //            col.Add(new Instruction(i.OpCode, newf));
        //            continue;
        //        }
        //        if (operand is TypeReference)
        //        {
        //            TypeReference tref = operand as TypeReference;
        //            TypeReference newf = targetModule.Import(tref);
        //            col.Add(new Instruction(i.OpCode, newf));
        //            continue;
        //        }
        //        if (operand is TypeDefinition)
        //        {
        //            TypeDefinition tdef = operand as TypeDefinition;
        //            TypeReference newf = targetModule.Import(tdef);
        //            col.Add(new Instruction(i.OpCode, newf));
        //            continue;
        //        }
        //        if (operand is MethodReference)
        //        {
        //            MethodReference mref = operand as MethodReference;
        //            MethodReference newf = targetModule.Import(mref);
        //            col.Add(new Instruction(i.OpCode, newf));
        //            continue;
        //        }

        //        // we don't need to do any processing on the operand
        //        col.Add(new Instruction(i.OpCode, operand));
        //    }

        //    // copy the exception handler blocks

        //    foreach (ExceptionHandler eh in oldBody.ExceptionHandlers)
        //    {
        //        ExceptionHandler neh = new ExceptionHandler(eh.HandlerType);
        //        neh.CatchType = targetModule.Import(eh.CatchType);

        //        // we need to setup neh.Start and End; these are instructions; we need to locate it in the source by index
        //        if (eh.TryStart != null)
        //        {
        //            int idx = oldBody.Instructions.IndexOf(eh.TryStart);
        //            neh.TryStart = col[idx];
        //        }
        //        if (eh.TryEnd != null)
        //        {
        //            int idx = oldBody.Instructions.IndexOf(eh.TryEnd);
        //            neh.TryEnd = col[idx];
        //        }

        //        nBody.ExceptionHandlers.Add(neh);
        //    }

        //    // Add this method to the target typedef
        //    copyToTypedef.Methods.Add(targetMethod);
        //    targetMethod.DeclaringType = copyToTypedef;
        //    return targetMethod;
        //}
    }
}
