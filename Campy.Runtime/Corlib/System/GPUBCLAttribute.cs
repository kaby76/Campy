using System;
using System.Collections.Generic;
using System.Text;

namespace System
{
    internal class GPUBCLAttribute : Attribute
    {
        public string _native_method_name;

        public GPUBCLAttribute(string native_method_name)
        {
            _native_method_name = native_method_name;
        }
    }
}
