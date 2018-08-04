namespace Campy.Meta
{
    using Mono.Cecil;
    using Mono.Collections.Generic;
    using System;
    using System.IO;

    public class StickyAssemblyResolver : Mono.Cecil.IAssemblyResolver
    {
        private DefaultAssemblyResolver resolver;
        private StickyMetadataResolver metadata_resolver;

        public StickyMetadataResolver MetadataResolver
        {
            get { return metadata_resolver; }
            set { metadata_resolver = value; }
        }

        public StickyAssemblyResolver()
        {
            resolver = new DefaultAssemblyResolver();
        }

        public void Dispose()
        {
            if (resolver != null)
                resolver.Dispose();
            resolver = null;
        }

        public AssemblyDefinition Resolve(AssemblyNameReference name)
        {
            // Instead of resolver.Resolve(name), add in parameters to again
            // override the metadata resolver.
            if (metadata_resolver == null) throw new Exception("Metadata resolver null.");
            return resolver.Resolve(name, new ReaderParameters { AssemblyResolver = this, MetadataResolver = metadata_resolver, ReadSymbols = false });
        }

        public AssemblyDefinition Resolve(AssemblyNameReference name, ReaderParameters parameters)
        {
            if (metadata_resolver == null) throw new Exception("Metadata resolver null.");
            return resolver.Resolve(name, parameters);
        }

        public void AddSearchDirectory(string path)
        {
            resolver.AddSearchDirectory(path);
        }
    }

    public class StickyMetadataResolver : Mono.Cecil.IMetadataResolver
    {
        private StickyAssemblyResolver assembly_resolver;

        public StickyAssemblyResolver AssemblyResolver
        {
            get { return assembly_resolver; }
            set { assembly_resolver = value; }
        }

        public StickyMetadataResolver(IAssemblyResolver assemblyResolver)
        {
            assembly_resolver = (StickyAssemblyResolver)assemblyResolver;
            assembly_resolver.MetadataResolver = this;
        }

        public virtual TypeDefinition Resolve(TypeReference type)
        {
            type = type.GetElementType();

            var scope = type.Scope;

            if (scope == null)
                return null;

            switch (scope.MetadataScopeType)
            {
                case MetadataScopeType.AssemblyNameReference:
                    var assembly = assembly_resolver.Resolve((AssemblyNameReference)scope);
                    if (assembly == null)
                        return null;

                    return GetType(assembly.MainModule, type);
                case MetadataScopeType.ModuleDefinition:
                    return GetType((ModuleDefinition)scope, type);
                case MetadataScopeType.ModuleReference:
                    var modules = type.Module.Assembly.Modules;
                    var module_ref = (ModuleReference)scope;
                    for (int i = 0; i < modules.Count; i++)
                    {
                        var netmodule = modules[i];
                        if (netmodule.Name == module_ref.Name)
                            return GetType(netmodule, type);
                    }
                    break;
            }

            throw new NotSupportedException();
        }

        static TypeDefinition GetType(ModuleDefinition module, TypeReference reference)
        {
            var type = GetTypeDefinition(module, reference);
            if (type != null)
                return type;

            if (!module.HasExportedTypes)
                return null;

            var exported_types = module.ExportedTypes;

            for (int i = 0; i < exported_types.Count; i++)
            {
                var exported_type = exported_types[i];
                if (exported_type.Name != reference.Name)
                    continue;

                if (exported_type.Namespace != reference.Namespace)
                    continue;

                return exported_type.Resolve();
            }

            return null;
        }

        static TypeDefinition GetTypeDefinition(ModuleDefinition module, TypeReference type)
        {
            if (!type.IsNested)
                return module.GetType(type.Namespace, type.Name);

            var declaring_type = type.DeclaringType.Resolve();
            if (declaring_type == null)
                return null;

            return declaring_type.GetNestedType(type.TypeFullName());
        }

        public virtual FieldDefinition Resolve(FieldReference field)
        {
            var type = Resolve(field.DeclaringType);
            if (type == null)
                return null;

            if (!type.HasFields)
                return null;

            return GetField(type, field);
        }

        FieldDefinition GetField(TypeDefinition type, FieldReference reference)
        {
            while (type != null)
            {
                var field = GetField(type.Fields, reference);
                if (field != null)
                    return field;

                if (type.BaseType == null)
                    return null;

                type = Resolve(type.BaseType);
            }

            return null;
        }

        static FieldDefinition GetField(Collection<FieldDefinition> fields, FieldReference reference)
        {
            for (int i = 0; i < fields.Count; i++)
            {
                var field = fields[i];

                if (field.Name != reference.Name)
                    continue;

                if (!AreSame(field.FieldType, reference.FieldType))
                    continue;

                return field;
            }

            return null;
        }

        public MethodDefinition Resolve(MethodReference method)
        {
            var type = Resolve(method.DeclaringType);
            if (type == null)
                return null;

            method = method.GetElementMethod();

            if (!type.HasMethods)
                throw new Exception("Metadata resolver--type has no methods!");

            var result = GetMethod(type, method);
            if (result != null)
            {
                if (result.Module.MetadataResolver != this)
                    throw new Exception("Can't be.");
                return result;
            }

            return result;
        }

        MethodDefinition GetMethod(TypeDefinition type, MethodReference reference)
        {
            while (type != null)
            {
                var result = GetMethod(type.Methods, reference);
                if (result != null)
                {
                    if (result != null)
                    {
                        if (result.Module.MetadataResolver != this)
                            throw new Exception("Can't be.");
                        return result;
                    }
                }

                if (type.BaseType == null)
                    return null;

                type = Resolve(type.BaseType);
                if (type != null)
                {
                    if (type.Module.MetadataResolver != this)
                        throw new Exception("Can't be.");
                    return result;
                }
            }

            return null;
        }

        public static MethodDefinition GetMethod(Collection<MethodDefinition> methods, MethodReference reference)
        {
            for (int i = 0; i < methods.Count; i++)
            {
                var method = methods[i];

                if (method.Name != reference.Name)
                    continue;

                if (method.HasGenericParameters != reference.HasGenericParameters)
                    continue;

                if (method.HasGenericParameters && method.GenericParameters.Count != reference.GenericParameters.Count)
                    continue;

                if (!AreSame(method.ReturnType, reference.ReturnType))
                    continue;

                if (method.IsVarArg() != reference.IsVarArg())
                    continue;

                if (method.IsVarArg() && IsVarArgCallTo(method, reference))
                    return method;

                if (method.HasParameters != reference.HasParameters)
                    continue;

                if (!method.HasParameters && !reference.HasParameters)
                    return method;

                if (!AreSame(method.Parameters, reference.Parameters))
                    continue;

                return method;
            }

            return null;
        }

        static bool AreSame(Collection<ParameterDefinition> a, Collection<ParameterDefinition> b)
        {
            var count = a.Count;

            if (count != b.Count)
                return false;

            if (count == 0)
                return true;

            for (int i = 0; i < count; i++)
                if (!AreSame(a[i].ParameterType, b[i].ParameterType))
                    return false;

            return true;
        }

        static bool IsVarArgCallTo(MethodDefinition method, MethodReference reference)
        {
            if (method.Parameters.Count >= reference.Parameters.Count)
                return false;

            if (reference.GetSentinelPosition() != method.Parameters.Count)
                return false;

            for (int i = 0; i < method.Parameters.Count; i++)
                if (!AreSame(method.Parameters[i].ParameterType, reference.Parameters[i].ParameterType))
                    return false;

            return true;
        }

        static bool AreSame(TypeSpecification a, TypeSpecification b)
        {
            if (!AreSame(a.ElementType, b.ElementType))
                return false;

            if (a.IsGenericInstance)
                return AreSame((GenericInstanceType)a, (GenericInstanceType)b);

            if (a.IsRequiredModifier || a.IsOptionalModifier)
                return AreSame((IModifierType)a, (IModifierType)b);

            if (a.IsArray)
                return AreSame((ArrayType)a, (ArrayType)b);

            return true;
        }

        static bool AreSame(ArrayType a, ArrayType b)
        {
            if (a.Rank != b.Rank)
                return false;

            // TODO: dimensions

            return true;
        }

        static bool AreSame(IModifierType a, IModifierType b)
        {
            return AreSame(a.ModifierType, b.ModifierType);
        }

        static bool AreSame(GenericInstanceType a, GenericInstanceType b)
        {
            if (a.GenericArguments.Count != b.GenericArguments.Count)
                return false;

            for (int i = 0; i < a.GenericArguments.Count; i++)
                if (!AreSame(a.GenericArguments[i], b.GenericArguments[i]))
                    return false;

            return true;
        }

        static bool AreSame(GenericParameter a, GenericParameter b)
        {
            return a.Position == b.Position;
        }

        static bool AreSame(GenericParameter a, TypeReference b)
        {
            if (ReferenceEquals(a, b))
                return true;

            if (a == null || b == null)
                return false;

            // etype is internal!
            //if (a.etype != b.etype)
            //    return false;

            if (b.IsGenericParameter)
                return AreSame((GenericParameter)a, (GenericParameter)b);

            //if (a.IsTypeSpecification())
            //    return AreSame((TypeSpecification)a, (TypeSpecification)b);

            // Since a generic parameter can unify with any type, return true
            return true;

            if (a.Name != b.Name || a.Namespace != b.Namespace)
                return false;

            //TODO: check scope

            return AreSame(a.DeclaringType, b.DeclaringType);
        }

        static bool AreSame(TypeReference a, GenericParameter b)
        {
            if (ReferenceEquals(a, b))
                return true;

            if (a == null || b == null)
                return false;

            // etype is internal!
            //if (a.etype != b.etype)
            //    return false;

            if (a.IsGenericParameter)
                return AreSame((GenericParameter)a, (GenericParameter)b);

            // if (a.IsTypeSpecification())
            //    return AreSame((TypeSpecification)a, (TypeSpecification)b);

            // Since a generic parameter can unify with any type, return true
            return true;

            if (a.Name != b.Name || a.Namespace != b.Namespace)
                return false;

            //TODO: check scope

            return AreSame(a.DeclaringType, b.DeclaringType);
        }

        static bool AreSame(TypeReference a, TypeReference b)
        {
            if (ReferenceEquals(a, b))
                return true;

            if (a == null || b == null)
                return false;

            //if (a.etype != b.etype)
            //    return false;

            if (a.IsGenericParameter && b.IsGenericParameter)
                return AreSame((GenericParameter)a, (GenericParameter)b);

            if (a.IsGenericParameter && !b.IsGenericParameter)
                return AreSame((GenericParameter)a, b);

            if (!a.IsGenericParameter && b.IsGenericParameter)
                return AreSame(a, (GenericParameter)b);

            if (a.IsTypeSpecification() && b.IsTypeSpecification())
                return AreSame((TypeSpecification)a, (TypeSpecification)b);

            if (a.Name != b.Name || a.Namespace != b.Namespace)
                return false;

            //TODO: check scope

            return AreSame(a.DeclaringType, b.DeclaringType);
        }
    }

    static class Mixin
    {
        public static TypeDefinition GetNestedType(this TypeDefinition self, string fullname)
        {
            if (!self.HasNestedTypes)
                return null;

            var nested_types = self.NestedTypes;

            for (int i = 0; i < nested_types.Count; i++)
            {
                var nested_type = nested_types[i];

                if (nested_type.TypeFullName() == fullname)
                    return nested_type;
            }

            return null;
        }

        public static bool IsVarArg(this IMethodSignature self)
        {
            return (self.CallingConvention & MethodCallingConvention.VarArg) != 0;
        }

        public static int GetSentinelPosition(this IMethodSignature self)
        {
            if (!self.HasParameters)
                return -1;

            var parameters = self.Parameters;
            for (int i = 0; i < parameters.Count; i++)
                if (parameters[i].ParameterType.IsSentinel)
                    return i;

            return -1;
        }

        public enum ElementType : byte
        {
            None = 0x00,
            Void = 0x01,
            Boolean = 0x02,
            Char = 0x03,
            I1 = 0x04,
            U1 = 0x05,
            I2 = 0x06,
            U2 = 0x07,
            I4 = 0x08,
            U4 = 0x09,
            I8 = 0x0a,
            U8 = 0x0b,
            R4 = 0x0c,
            R8 = 0x0d,
            String = 0x0e,
            Ptr = 0x0f,   // Followed by <type> token
            ByRef = 0x10,   // Followed by <type> token
            ValueType = 0x11,   // Followed by <type> token
            Class = 0x12,   // Followed by <type> token
            Var = 0x13,   // Followed by generic parameter number
            Array = 0x14,   // <type> <rank> <boundsCount> <bound1>  <loCount> <lo1>
            GenericInst = 0x15,   // <type> <type-arg-count> <type-1> ... <type-n> */
            TypedByRef = 0x16,
            I = 0x18,   // System.IntPtr
            U = 0x19,   // System.UIntPtr
            FnPtr = 0x1b,   // Followed by full method signature
            Object = 0x1c,   // System.Object
            SzArray = 0x1d,   // Single-dim array with 0 lower bound
            MVar = 0x1e,   // Followed by generic parameter number
            CModReqD = 0x1f,   // Required modifier : followed by a TypeDef or TypeRef token
            CModOpt = 0x20,   // Optional modifier : followed by a TypeDef or TypeRef token
            Internal = 0x21,   // Implemented within the CLI
            Modifier = 0x40,   // Or'd with following element types
            Sentinel = 0x41,   // Sentinel for varargs method signature
            Pinned = 0x45,   // Denotes a local variable that points at a pinned object

            // special undocumented constants
            Type = 0x50,
            Boxed = 0x51,
            Enum = 0x55
        }

        public static bool IsPrimitive(this ElementType self)
        {
            switch (self)
            {
                case ElementType.Boolean:
                case ElementType.Char:
                case ElementType.I:
                case ElementType.U:
                case ElementType.I1:
                case ElementType.U1:
                case ElementType.I2:
                case ElementType.U2:
                case ElementType.I4:
                case ElementType.U4:
                case ElementType.I8:
                case ElementType.U8:
                case ElementType.R4:
                case ElementType.R8:
                    return true;
                default:
                    return false;
            }
        }

        public static string TypeFullName(this TypeReference self)
        {
            return string.IsNullOrEmpty(self.Namespace)
                ? self.Name
                : self.Namespace + '.' + self.Name;
        }

        public static bool IsTypeOf(this TypeReference self, string @namespace, string name)
        {
            return self.Name == name
                && self.Namespace == @namespace;
        }

        public static bool IsTypeSpecification(this TypeReference type)
        {
            var predicate = type as TypeSpecification != null;
            return predicate;

            //switch (type.etype)
            //{
            //    case ElementType.Array:
            //    case ElementType.ByRef:
            //    case ElementType.CModOpt:
            //    case ElementType.CModReqD:
            //    case ElementType.FnPtr:
            //    case ElementType.GenericInst:
            //    case ElementType.MVar:
            //    case ElementType.Pinned:
            //    case ElementType.Ptr:
            //    case ElementType.SzArray:
            //    case ElementType.Sentinel:
            //    case ElementType.Var:
            //        return true;
            //}
            //return false;
        }

        public static TypeDefinition CheckedResolve(this TypeReference self)
        {
            var type = self.Resolve();
            if (type == null)
                throw new ResolutionException(self);

            return type;
        }
    }

    public class StickyReadMod
    {
        public static ModuleDefinition StickyReadModule(string fileName)
        {
            // Warning.
            // Verify that there are no calls to Mono.Cecil.ModuleDefinition.ReadModule(fileName);
            // in Campy!
            string p = Path.GetDirectoryName(fileName);
            // There is no ReaderParameter set. Create one.
            var resolver = new StickyAssemblyResolver();
            resolver.AddSearchDirectory(p);
            var metadata_resolver = new StickyMetadataResolver(resolver);
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(
                fileName,
                new ReaderParameters { AssemblyResolver = resolver, MetadataResolver = metadata_resolver });
            return md;
        }

        public static ModuleDefinition StickyReadModule(string fileName, ReaderParameters parameters)
        {
            StickyAssemblyResolver resolver = null;
            if (parameters.AssemblyResolver == null)
            {
                resolver = new StickyAssemblyResolver();
            }
            else if (parameters.AssemblyResolver as StickyAssemblyResolver == null)
                throw new Exception("You must use StickyAssemblyResolver in StickyReadModule.");
            else
                resolver = (StickyAssemblyResolver)parameters.AssemblyResolver;

            string p = Path.GetDirectoryName(fileName);
            resolver.AddSearchDirectory(p);


            StickyMetadataResolver metadata_resolver = null;
            if (parameters.MetadataResolver == null)
                metadata_resolver = new StickyMetadataResolver(resolver);
            else if (parameters.MetadataResolver as StickyMetadataResolver == null)
                throw new Exception("You must use StickyMetadataResolver in StickyReadModule.");
            else
                metadata_resolver = (StickyMetadataResolver)parameters.MetadataResolver;

            parameters.AssemblyResolver = resolver;
            parameters.MetadataResolver = metadata_resolver;

            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(fileName, parameters);
            return md;
        }
    }
}
