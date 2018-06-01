using System;
using System.Collections.Generic;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;
using Mono.Collections.Generic;

namespace Campy.Utils
{
    public static class Class1
    {
        // Create a type definition corresponding to a generic instance type. The generic instance type is
        // first created using the generic type and generic arguments. Everything about the original type
        // must be added to the new type definition.
        public static TypeReference ConvertGenericInstanceTypeToNonGenericInstanceType(this TypeReference type)
        {
            // Verify that type is generic instance.
            if (type as GenericInstanceType == null)
                return type;

            // Get generic arguments and pass to method that does construction.
            var args = (type as GenericInstanceType).GenericArguments.ToArray();

            var uninstance = type.Resolve();

            // Find if type has been created before.
            var f = uninstance.Module.GetType(type.FullName);
            if (f != null)
                return f;
            var any = _cache.Find(i => i.FullName == type.FullName);
            if (any != null)
                return any;

            var m = MakeGenericInstanceTypeDefintionAux(uninstance, args);

            var bb = type.Module.ImportReference(m);

            return m;
        }

        private static List<TypeDefinition> _cache = new List<TypeDefinition>();

        private static TypeReference ConvertGenericParameterToTypeReference(TypeReference type,
            params TypeReference[] generic_arguments)
        {
            if (type as GenericParameter != null)
            {
                var gp = type as GenericParameter;
                var num = gp.Position;
                var yo = generic_arguments.ToArray()[num];
                type = yo;
            }
            if (type.IsArray)
            {
                var array_type = type as Mono.Cecil.ArrayType;
                var element_type = array_type.ElementType;
                var new_element_type = ConvertGenericParameterToTypeReference(element_type, generic_arguments);
                if (element_type != new_element_type)
                {
                    var new_array_type = new ArrayType(new_element_type,
                        array_type.Rank);
                    type = new_array_type;
                }
            }

            if (type as GenericInstanceType != null)
            {
                // For generic instance types, it could contain a generic parameter.
                // Substitute parameter if needed.
                var git = type as GenericInstanceType;
                var args = git.GenericArguments;
                var new_args = git.GenericArguments.ToArray();
                for (int i = 0; i < new_args.Length; ++i)
                {
                    var arg = args[i];
                    var new_arg = ConvertGenericParameterToTypeReference(arg, generic_arguments);
                    git.GenericArguments[i] = new_arg;
                }
            }
            return type;
        }

        public static TypeDefinition MakeGenericInstanceTypeDefintionAux(TypeReference type, params TypeReference[] generic_arguments)
        {
            if (type.GenericParameters.Count != generic_arguments.Length)
                throw new ArgumentException();

            var instance = new GenericInstanceType(type);
            foreach (var argument in generic_arguments)
                instance.GenericArguments.Add(argument);

            string name = instance.FullName.Substring(instance.Namespace.Length + 1);
            TypeAttributes ta = type.Resolve().Attributes;

            // First create the type definition.
            TypeDefinition new_definition = new TypeDefinition(
                instance.Namespace,
                name,
                ta, type.DeclaringType);

            // Cache for uses that may pop up.
            _cache.Add(new_definition);

            // Add in all fields.
            var fields = type.Resolve().Fields;
            for (int i = 0; i < fields.Count; ++i)
            {
                var field = fields[i];
                var field_definition = field as FieldDefinition;
                var field_type = field_definition.FieldType;
                var new_field_type = ConvertGenericParameterToTypeReference(field_type, generic_arguments);
                var new_field_definition = new FieldDefinition(field_definition.Name,
                        field_definition.Attributes,
                        new_field_type);
                new_field_definition.DeclaringType = new_definition;
                new_definition.Fields.Insert(i, new_field_definition);
            }

            // Add in all properties.
            var properties = type.Resolve().Properties;
            for (int i = 0; i < properties.Count; ++i)
            {
                var property = properties[i];
                var property_definition = property as PropertyDefinition;
                var vv = property_definition.PropertyType as GenericParameter;
                var new_property_type = property_definition.PropertyType;
                new_property_type = ConvertGenericParameterToTypeReference(new_property_type, generic_arguments);
                var new_property_definition = new PropertyDefinition(property_definition.Name,
                    property_definition.Attributes,
                    new_property_type);
                new_property_definition.DeclaringType = new_definition;
                new_definition.Properties.Insert(i, new_property_definition);
            }

            // Add in all methods.
            var methods = type.Resolve().Methods;
            for (int i = 0; i < methods.Count; ++i)
            {
                var method = methods[i];
                var method_definition = method as MethodDefinition;

                var ret = method_definition.ReturnType as GenericParameter;
                var new_ret_type = method_definition.ReturnType;
                new_ret_type = ConvertGenericParameterToTypeReference(new_ret_type, generic_arguments);
                var new_method_definition = new MethodDefinition(method_definition.Name,
                    method_definition.Attributes,
                    new_definition);
                new_method_definition.ReturnType = new_ret_type;
                new_method_definition.HasThis = method_definition.HasThis;
                foreach (var param in method_definition.Parameters)
                {
                    var parameter_type = param.ParameterType;
                    var new_parameter_type = ConvertGenericParameterToTypeReference(parameter_type, generic_arguments);
                    new_parameter_type = new_parameter_type.ConvertGenericInstanceTypeToNonGenericInstanceType();
                    var new_param = new ParameterDefinition(
                        param.Name,
                        param.Attributes,
                        new_parameter_type);
                    new_method_definition.Parameters.Add(new_param);
                }

                new_definition.Methods.Insert(i, new_method_definition);
                var new_body = new_method_definition.Body;
                var body = method_definition.Body;
                if (method_definition.Body != null)
                {
                    var worker = new_body.GetILProcessor();
                    for (int j = 0; j < body.Instructions.Count; ++j)
                    {
                        Instruction n = body.Instructions[j];
                        object operand = n.Operand;
                        Instruction new_inst = n;

                        var operand_type_reference = operand as TypeReference;
                        if (operand_type_reference != null)
                        {
                            var tr = operand as TypeReference;
                            var new_tr = ConvertGenericParameterToTypeReference(tr, generic_arguments);
                            new_tr = new_tr.ConvertGenericInstanceTypeToNonGenericInstanceType();
                            // fix instruction.
                            new_inst.Operand = new_tr;
                        }

                        if (operand as FieldReference != null)
                        {
                            var c1 = operand as FieldReference;
                            var c2 = c1.FieldType;
                            if (c2 != null)
                            {
                                var new_c1_declaring_type = ConvertGenericParameterToTypeReference(c1.DeclaringType, generic_arguments);
                                var new_c2 = ConvertGenericParameterToTypeReference(c2, generic_arguments);
                                new_c2 = new_c2.ConvertGenericInstanceTypeToNonGenericInstanceType();

                                var yofields = method_definition.DeclaringType.Fields;
                                c1 = new FieldReference(c1.Name, new_c2);
                                c1.DeclaringType = new_c1_declaring_type;
                                new_inst = worker.Create(n.OpCode, c1);
                                new_inst.Offset = n.Offset;
                            }
                        }

                        new_method_definition.Body.Instructions.Insert(j, new_inst);
                    }
                }
            }

            type.Module.Types.Add(new_definition);
            return new_definition;
        }

        public static TypeReference InstantiateGenericTypeReference(TypeReference type)
        {
            TypeReference result = type;

            if (type.IsGenericInstance)
            {
                // Create non-generic type out of a generic type instance.
                var git = type as GenericInstanceType;
                result = git.ConvertGenericInstanceTypeToNonGenericInstanceType();
            }

            return result;
        }

        public static MethodReference MakeMethodReference(this MethodDefinition method)
        {
            var reference = new MethodReference(method.Name, method.ReturnType, method.DeclaringType);

            foreach (ParameterDefinition parameter in method.Parameters)
                reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));
            return reference;
        }

        public static MethodReference MakeMethodReference(this MethodReference method, TypeReference declaringType)
        {
            var reference = new MethodReference(method.Name, method.ReturnType, declaringType);

            foreach (ParameterDefinition parameter in method.Parameters)
                reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));
            return reference;
        }

        public static TypeReference MakeGenericType(TypeReference type, params
            TypeReference[] arguments)
        {
            if (type.GenericParameters.Count != arguments.Length)
                throw new ArgumentException();

            var instance = new GenericInstanceType(type);
            foreach (var argument in arguments)
                instance.GenericArguments.Add(argument);

            return instance;
        }

        public static MethodReference MakeHostInstanceGeneric(
            MethodReference self,
            params TypeReference[] args)
        {
            var reference = new MethodReference(
                self.Name,
                self.ReturnType,
                self.DeclaringType.MakeGenericInstanceType(args))
            {
                HasThis = self.HasThis,
                ExplicitThis = self.ExplicitThis,
                CallingConvention = self.CallingConvention
            };

            foreach (var parameter in self.Parameters)
                reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));

            foreach (var genericParam in self.GenericParameters)
                reference.GenericParameters.Add(new GenericParameter(genericParam.Name, reference));

            return reference;
        }
    }
}
