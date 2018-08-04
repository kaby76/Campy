using System;
using System.Collections.Generic;
using System.Text;

namespace Campy.Compiler
{
    public class SafeStackQueue<T> : Campy.Utils.StackQueue<T> where T : Mono.Cecil.TypeReference
    {
        public override void Push(T value)
        {
            if (value != null && value.ContainsGenericParameter)
                throw new Exception("Pushing value that contains generic.");
            if (value != null && value.FullName.Contains("`") && !value.FullName.Contains("<"))
                throw new Exception("Pushing value that contains faulty generic.");
            base.Push(value);
        }
    }
}
