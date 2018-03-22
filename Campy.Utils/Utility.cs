using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using System.IO;
using Mono.Cecil;

namespace Campy.Utils
{
    public class Utility
    {

        public static string FindExePath(string exe)
        {
            exe = Environment.ExpandEnvironmentVariables(exe);
            if (!File.Exists(exe))
            {
                if (Path.GetDirectoryName(exe) == String.Empty)
                {
                    foreach (string test in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';'))
                    {
                        string path = test.Trim();
                        if (!String.IsNullOrEmpty(path) && File.Exists(path = Path.Combine(path, exe)))
                            return Path.GetFullPath(path);
                    }
                }
                return null;
            }
            return Path.GetFullPath(exe);
        }

        /// <summary>
        /// C# .NET really does not provide any API to get a "user friendly" name of a type,
        /// especially generics. The function Simplify and GetFriendlyTypeName fill in that gap.
        /// The prefix for a type, e.g., "System.Collections." is removed only if it is in the
        /// System namespace.
        /// </summary>
        /// <param name="str"></param>
        /// <returns></returns>
        public static string Simplify(String str)
        {
            if (str.Equals("System.Boolean"))
                return "bool";
            if (str.Equals("System.Byte"))
                return "byte";
            if (str.Equals("System.Char"))
                return "char";
            if (str.Equals("System.Decimal"))
                return "decimal";
            if (str.Equals("System.Double"))
                return "double";
            if (str.Equals("System.Single"))
                return "float";
            if (str.Equals("System.Int32"))
                return "int";
            if (str.Equals("System.Int64"))
                return "long";
            if (str.Equals("System.SByte"))
                return "sbyte";
            if (str.Equals("System.Int16"))
                return "short";
            if (str.Equals("System.UInt32"))
                return "uint";
            if (str.Equals("System.UInt64"))
                return "ulong";
            if (str.Equals("System.UInt16"))
                return "ushort";
            if (str.Equals("System.Void"))
                return "void";

            //            if (str.IndexOf("System.") == 0)
            //               return str.Substring(1 + str.LastIndexOf("."));
            //           if (str.IndexOf("Campy.Types.") == 0)
            //                return str.Substring(1 + str.LastIndexOf("."));
            str = str.Replace('+', '.');
            return str;
        }

        /// <summary>
        /// C# .NET really does not provide any API to get a "user friendly" name of a type,
        /// especially generics. The function Simplify and GetFriendlyTypeName fill in that gap.
        /// The prefix for a type, e.g., "System.Collections." is removed only if it is in the
        /// System namespace.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string GetFriendlyTypeName(Type type)
        {
            if (type.IsGenericParameter)
            {
                return Simplify(type.Name);
            }

            if (!type.IsGenericType)
            {
                return Simplify(type.FullName);
            }

            StringBuilder builder = new StringBuilder();
            String name = Simplify(type.Name);

            // If generic, then a backtick occurs in the name. In that case, remove the trailing information.
            String pre;
            int index = name.IndexOf("`");
            if (index >= 0)
                pre = String.Format("{0}.{1}", type.Namespace, Simplify(name.Substring(0, index)));
            else
                pre = String.Format("{0}.{1}", type.Namespace, Simplify(name));
            pre = Simplify(pre);
            builder.Append(pre);
            builder.Append('<');
            bool first = true;
            foreach (Type arg in type.GetGenericArguments())
            {
                if (!first)
                {
                    builder.Append(',');
                }
                builder.Append(GetFriendlyTypeName(arg));
                first = false;
            }
            builder.Append('>');
            // Convert "+" signs into "." since it's just a nested class.
            String result = builder.ToString();
            result = result.Replace('+', '.');
            return result;
        }

        public static string GetFriendlyTypeNameMono(Mono.Cecil.TypeReference type)
        {
            if (type.IsGenericParameter)
            {
                return Simplify(type.Name);
            }

            if (!type.HasGenericParameters)
            {
                return Simplify(type.FullName);
            }

            StringBuilder builder = new StringBuilder();
            String name = Simplify(type.Name);

            // If generic, then a backtick occurs in the name. In that case, remove the trailing information.
            String pre;
            int index = name.IndexOf("`");
            if (index >= 0)
                pre = String.Format("{0}.{1}", type.Namespace, Simplify(name.Substring(0, index)));
            else
                pre = String.Format("{0}.{1}", type.Namespace, Simplify(name));
            pre = Simplify(pre);
            builder.Append(pre);
            builder.Append('<');
            bool first = true;
            foreach (Mono.Cecil.GenericParameter arg in type.GenericParameters)
            {
                if (!first)
                {
                    builder.Append(',');
                }
                builder.Append(GetFriendlyTypeNameMono(arg));
                first = false;
            }
            builder.Append('>');
            // Convert "+" signs into "." since it's just a nested class.
            String result = builder.ToString();
            result = result.Replace('+', '.');
            return result;
        }

        public static String RemoveGenericParameters(Type type)
        {
            String result = type.FullName;
            result = Regex.Replace(result, "\\[\\[.*\\]\\]", "");
            return result;
        }

        /// <summary>
        /// This function is a mostly a hack. It substitutes underscores for any characters
        /// in a string so that it can be used as a variable in C++ AMP code, and for file naming.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static String NormalizeSystemReflectionName(String name)
        {
            String result = name;
            result = result.Replace(" ", "_");
            result = result.Replace("<", "_");
            result = result.Replace(">", "_");
            result = result.Replace("::", "_");
            result = result.Replace("/", "_");
            result = result.Replace("+", "_");
            result = result.Replace("(", "_");
            result = result.Replace(")", "_");
            result = result.Replace("[", "_");
            result = result.Replace("]", "_");
            result = result.Replace(".", "_");
            result = result.Replace(",", "_");
            result = result.Replace("`", "_");
            return result;
        }

        public static String NormalizeMonoCecilName(String name)
        {
            String result = name;
            result = result.Replace(" ", "_");
            result = result.Replace("<", "_");
            result = result.Replace(">", "_");
            result = result.Replace("::", "_");
            result = result.Replace("/", "_");
            result = result.Replace("+", "_");
            result = result.Replace("(", "_");
            result = result.Replace(")", "_");
            result = result.Replace("[", "_");
            result = result.Replace("]", "_");
            result = result.Replace(".", "_");
            result = result.Replace(",", "_");
            result = result.Replace("`", "_");
            return result;
        }


        /// <summary>
        /// Special type used to match any generic parameter type in GetMethodExt().
        /// </summary>
        public class T
        { }

        static Dictionary<Type, Type> matches = new Dictionary<Type, Type>();

        private static bool IsUnifiable(Type t1, Type t2)
        {
            // If both are not generic then they have to be equal types.
            if ((!t1.IsGenericParameter) && (!t2.IsGenericParameter))
                return t1 == t2;

            // Handle any generic arguments
            if (t1.IsGenericType && t2.IsGenericType)
            {
                Type[] t1Arguments = t1.GetGenericArguments();
                Type[] t2Arguments = t2.GetGenericArguments();
                if (t1Arguments.Length == t2Arguments.Length)
                {
                    for (int i = 0; i < t1Arguments.Length; ++i)
                    {
                        if (!IsSimilarType(t1Arguments[i], t2Arguments[i]))
                            return false;
                    }
                    return true;
                }
            }

            // Find if matches for either t1 or t2 in matching table.
            Type match_t1 = null;
            Type match_t2 = null;
            try
            {
                match_t1 = matches[t1];
            }
            catch
            {
            }
            try
            {
                match_t2 = matches[t2];
            }
            catch
            {
            }

            // If entry for match for either, then the match has to match.
            if (match_t1 != null)
                return match_t1 == t2;
            if (match_t2 != null)
                return match_t2 == t1;

            // Not matched before, so these match.
            matches.Add(t1, t2);
            matches.Add(t2, t1);

            return true;
        }

        static Dictionary<Type, Mono.Cecil.TypeReference> matches_mono = new Dictionary<Type, Mono.Cecil.TypeReference>();

        private static bool IsUnifiableMono(Type t1, Mono.Cecil.TypeReference t2)
        {
            // If both are not generic then they have to be equal types.
            if ((!t1.IsGenericParameter) && (!t2.IsGenericParameter))
                return t1.Name == t2.Name;

            // Handle any generic arguments
            if (t1.IsGenericType && t2.HasGenericParameters)
            {
                Type[] t1Arguments = t1.GetGenericArguments();
                Mono.Collections.Generic.Collection<Mono.Cecil.GenericParameter> t2Arguments = t2.GenericParameters;
                if (t1Arguments.Length == t2Arguments.Count)
                {
                    for (int i = 0; i < t1Arguments.Length; ++i)
                    {
                        if (!IsSimilarType(t1Arguments[i], t2Arguments[i]))
                            return false;
                    }
                    return true;
                }
            }

            // Find if matches for either t1 or t2 in matching table.
            Mono.Cecil.TypeReference match_t1 = null;
            try
            {
                match_t1 = matches_mono[t1];
            }
            catch
            {
            }

            // If entry for match for either, then the match has to match.
            if (match_t1 != null)
                return match_t1 == t2;

            // Not matched before, so these match.
            matches_mono.Add(t1, t2);

            return true;
        }

        public static void Reset()
        {
            matches = new Dictionary<Type, Type>();
        }

        /// <summary>
        /// Determines if the two types are either identical, or are both generic 
        /// parameters or generic types with generic parameters in the same
        ///  locations (generic parameters match any other generic paramter,
        /// but NOT concrete types).
        /// </summary>
        public static bool IsSimilarType(Type thisType, Type type)
        {
            // Ignore any 'ref' types
            if (thisType.IsByRef)
                thisType = thisType.GetElementType();
            if (type.IsByRef)
                type = type.GetElementType();

            // Handle array types
            if (thisType.IsArray && type.IsArray)
            {
                // Dimensions must be the same.
                if (thisType.GetArrayRank() != type.GetArrayRank())
                    return false;
                // Base type of array must be the same.
                return IsSimilarType(thisType.GetElementType(), type.GetElementType());
            }
            if (thisType.IsArray && !type.IsArray)
                return false;
            if (type.IsArray && !thisType.IsArray)
                return false;

            // If the types are identical, or they're both generic parameters 
            // or the special 'T' type, treat as a match
            // Match also if thisType is generic and type can be unified with thisType.
            if (thisType == type // identical types.
                || ((thisType.IsGenericParameter || thisType == typeof(T)) && (type.IsGenericParameter || type == typeof(T))) // using "T" as matching generic type.
                || IsUnifiable(thisType, type))
                return true;

            return false;
        }

        public static bool IsSimilarType(Type thisType, Mono.Cecil.TypeReference type)
        {
            // Ignore any 'ref' types
            if (thisType.IsByRef)
                thisType = thisType.GetElementType();
            if (type.IsByReference)
            {
                if (type.IsArray)
                {
                    var array_type = type as ArrayType;
                    type = array_type.ElementType;
                }
                else
                {
                    type = type.GetElementType();
                }
            }

            // Handle array types
            if (thisType.IsArray && type.IsArray)
            {
                Mono.Cecil.ArrayType at = type as Mono.Cecil.ArrayType;
                // Dimensions must be the same.
                if (thisType.GetArrayRank() != at.Rank)
                    return false;
                // Base type of array must be the same.
                var array_type = type as ArrayType;
                return IsSimilarType(thisType.GetElementType(), array_type.ElementType);
            }
            if (thisType.IsArray && !type.IsArray)
                return false;
            if (type.IsArray && !thisType.IsArray)
                return false;

            // If the types are identical, or they're both generic parameters 
            // or the special 'T' type, treat as a match
            // Match also if thisType is generic and type can be unified with thisType.
            if (thisType.Name == type.Name // identical types.
                || ((thisType.IsGenericParameter || thisType == typeof(T)) && (type.IsGenericParameter || type.Name.Equals("T"))) // using "T" as matching generic type.
                || IsUnifiableMono(thisType, type))
                return true;

            return false;
        }

    }

}
