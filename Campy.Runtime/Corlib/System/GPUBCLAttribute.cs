using System;
using System.Collections.Generic;
using System.Text;

namespace System
{
    internal class GPUBCLAttribute : Attribute
    {
        public string _native_method_name;
        public string _mangled_native_method_name;

        public GPUBCLAttribute(string native_method_name, string mangled_native_method_name)
        {
            _native_method_name = native_method_name;
            _mangled_native_method_name = mangled_native_method_name;
        }
    }
}
