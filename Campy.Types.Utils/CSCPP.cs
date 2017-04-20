using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Reflection;

namespace Campy.Types.Utils
{
    public class CSCPP
    {
        public static String ConvertToCPPCLIWithNameSpace(Type type, int level)
        {
            String result = "";

            // Use reflection to create atring equivalent to type in C++ unmanaged world.
            if (type.FullName.Equals("System.Int32"))
            {
                return "int";
            }
            else if (type.FullName.Equals("System.UInt32"))
            {
                return "unsigned int";
            }
            else if (type.IsClass || Campy.Types.Utils.ReflectionCecilInterop.IsStruct(type))
            {
                // Emit namespace declarations.
                if (level == 0)
                {
                    Type declaring_type = type.DeclaringType;
                    result += @"

#pragma once

";

                    String[] prefix = type.Namespace.Split(new char[] { '.' });
                    foreach (String p in prefix)
                    {
                        result += "namespace " + p + "{\n";
                    }
                }

                String ind = "";
                for (int i = 0; i < level; ++i)
                    ind += "    ";
                result += ind + "public ref class " + type.Name + "\n";
                result += ind + "{\n";
                BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic |
                    BindingFlags.Static | BindingFlags.Instance |
                    BindingFlags.DeclaredOnly;
                FieldInfo[] fields = type.GetFields(flags);
                for (int i = 0; i < fields.Length; ++i)
                {
                    FieldInfo fi = fields[i];
                    Type tf = fi.FieldType;
                    result += ConvertToCPPCLI(tf, level + 1) + " " + fi.Name + ";\n";
                }
                result += ind + "};\n";

                if (level == 0)
                {
                    String[] prefix = type.Namespace.Split(new char[] { '.' });
                    foreach (String p in prefix)
                    {
                        result += "}\n";
                    }

                    result += @"

";

                }

                return result;
            }
            else return null;
        }

        public static String ConvertToCPPWithNameSpace(Type type, int level)
        {
            String result = "";

            // Use reflection to create atring equivalent to type in C++ unmanaged world.
            if (type.FullName.Equals("System.Int32"))
            {
                return "int";
            }
            else if (type.FullName.Equals("System.UInt32"))
            {
                return "unsigned int";
            }
            else if (type.IsClass || Campy.Types.Utils.ReflectionCecilInterop.IsStruct(type))
            {
                // Emit namespace declarations.
                if (level == 0)
                {
                    Type declaring_type = type.DeclaringType;
                    result += @"

#pragma once

#pragma managed(push, off)

";

                    String[] prefix = type.Namespace.Split(new char[] { '.' });
                    foreach (String p in prefix)
                    {
                        result += "namespace " + p + "{\n";
                    }
                }

                String ind = "";
                for (int i = 0; i < level; ++i)
                    ind += "    ";
                result += ind + "struct " + type.Name + "\n";
                result += ind + "{\n";
                BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic |
                    BindingFlags.Static | BindingFlags.Instance |
                    BindingFlags.DeclaredOnly;
                FieldInfo[] fields = type.GetFields(flags);
                for (int i = 0; i < fields.Length; ++i)
                {
                    FieldInfo fi = fields[i];
                    Type tf = fi.FieldType;
                    result += ConvertToCPPCLI(tf, level + 1) + " " + fi.Name + ";\n";
                }
                result += ind + "};\n";

                if (level == 0)
                {
                    String[] prefix = type.Namespace.Split(new char[] { '.' });
                    foreach (String p in prefix)
                    {
                        result += "}\n";
                    }

                    result += @"

#pragma managed(pop)
";

                }

                return result;
            }
            else return null;
        }

        public static String ConvertToCPP(Type type, int level)
        {
            // Use reflection to create atring equivalent to type in C++ unmanaged world.
            if (type.FullName.Equals("System.Int32"))
            {
                return "int";
            }
            else if (type.FullName.Equals("System.UInt32"))
            {
                return "unsigned int";
            }
            else if (type.IsClass || Campy.Types.Utils.ReflectionCecilInterop.IsStruct(type))
            {
                // Complex type.
                String result = "";
                String ind = "";
                for (int i = 0; i < level; ++i)
                    ind += "    ";
                result += ind + "struct " + type.Name + "\n";
                result += ind + "{\n";
                BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic |
                    BindingFlags.Static | BindingFlags.Instance |
                    BindingFlags.DeclaredOnly;
                FieldInfo[] fields = type.GetFields(flags);
                for (int i = 0; i < fields.Length; ++i)
                {
                    FieldInfo fi = fields[i];
                    Type tf = fi.FieldType;
                    result += ConvertToCPPCLI(tf, level + 1) + " " + fi.Name + ";\n";
                }
                result += ind + "};\n";
                return result;
            }
            else return null;
        }

		public static String ConvertToCPPCLI(Type type, int level)
		{
			// Use reflection to create atring equivalent to type in C++ unmanaged world.
			if (type.FullName.Equals("System.Int32"))
			{
				return "int";
			}
			else if (type.FullName.Equals("System.UInt32"))
			{
				return "unsigned int";
			}
			else if (type.FullName.Equals("System.Single"))
			{
				return "float";
			}
			else if (!type.IsValueType)
			{
				// Complex type.
				String result = "";
				String ind = "";
				for (int i = 0; i < level; ++i)
					ind += "    ";
				result += ind + "ref struct " + type.Name + "\n";
				result += ind + "{\n";
                BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic |
					BindingFlags.Static | BindingFlags.Instance |
					BindingFlags.DeclaredOnly;
				FieldInfo[] fields = type.GetFields(flags);
				for (int i = 0; i < fields.Length; ++i)
				{
					FieldInfo fi = fields[i];
					Type tf = fi.FieldType;
					result += ConvertToCPPCLI(tf, level + 1) + " " + fi.Name + ";\n";
				}
				result += ind + "} " + type.Name + ";\n";
				return result;
			}
			else return null;
		}
	}
}


