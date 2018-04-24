using System;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Collections.Generic;

namespace Campy.Utils
{
    public static class Class1
    {
        // Create a type definition corresponding to a generic instance type. The generic instance type is
        // first created using the generic type and generic arguments. Everything about the original type
        // must be added to the new type definition.
        public static TypeDefinition MakeGenericInstanceTypeDefinition(this TypeReference type)
        {
            // Verify that type is generic instance.
            if (type as GenericInstanceType == null)
                throw new Exception("Trying to define type for a non-generic instance type.");

            // Get generic arguments and pass to method that does construction.
            var args = (type as GenericInstanceType).GenericArguments.ToArray();

            // Find if type has been created before.
            var t = type;
            var f = t.Module.GetType(type.FullName);

            var uninstance = type.Resolve();

            var m = MakeGenericInstanceTypeDefintionAux(uninstance, args);

            // Check: bb should be a fully instantiated type and have methods with bodies.
            var bb = t.Module.ImportReference(m);
            TypeDefinition ee = bb.Resolve();

            return m;
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

            // Add in all fields.
            var fields = type.Resolve().Fields;
            for (int i = 0; i < fields.Count; ++i)
            {
                var field = fields[i];
                var field_definition = field as FieldDefinition;
                var vv = field_definition.FieldType as GenericParameter;
                var new_field_type = field_definition.FieldType;
                if (vv != null)
                {
                    var gp = new_field_type as GenericParameter;
                    var num = gp.Position;
                    var yo = generic_arguments.ToArray()[num];
                    new_field_type = yo;
                }
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
                if (vv != null)
                {
                    var gp = new_property_type as GenericParameter;
                    var num = gp.Position;
                    var yo = generic_arguments.ToArray()[num];
                    new_property_type = yo;
                }
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
                if (ret != null)
                {
                    var gp = new_ret_type as GenericParameter;
                    var num = gp.Position;
                    var yo = generic_arguments.ToArray()[num];
                    new_ret_type = yo;
                }

                var new_method_definition = new MethodDefinition(method_definition.Name,
                    method_definition.Attributes,
                    new_definition);
                new_method_definition.ReturnType = new_ret_type;
                new_definition.Methods.Insert(i, new_method_definition);
                var new_body = new_method_definition.Body;
                var body = method_definition.Body;
                var worker = new_body.GetILProcessor();
                for (int j = 0; j < body.Instructions.Count; ++j)
                {
                    Instruction n = body.Instructions[j];
                    object operand = n.Operand;
                    Instruction new_inst = n;
                    if (operand as GenericParameter != null)
                    {
                        var c1 = operand as GenericParameter;
                        var num = c1.Position;
                        var yo = generic_arguments.ToArray()[num];
                        new_inst = worker.Create(n.OpCode, yo);
                        new_inst.Offset = n.Offset;
                    }
                    else if (operand as FieldReference != null)
                    {
                        var c1 = operand as FieldReference;
                        var c2 = c1.FieldType;
                        if (c2 != null)
                        {
                            if (c2 as GenericParameter != null)
                            {
                                var c3 = c2 as GenericParameter;
                                var num = c3.Position;
                                var yo = generic_arguments.ToArray()[num];
                                // Note, "Name" of the field needs adjusting due to that it's part of a generic instance type.
                                // For now, sluff.
                                string mname = c1.Name;
                                var yofields = method_definition.DeclaringType.Fields;
                                FieldDefinition fd = new_definition.DeclaringType.Fields.First(f => f.Name == mname);
                                c1 = new FieldReference(name, yo);
                                new_inst = worker.Create(n.OpCode, c1);
                            }
                        }
                    }
                    new_method_definition.Body.Instructions.Insert(j, new_inst);
                }
            }

            type.Module.Types.Add(new_definition);
            return new_definition;
        }

        public static MethodDefinition MakeGenericMethod(MethodReference method, TypeReference declaringType)
        {
            var md = method.Module;
            var generic_method = md.ImportReference(method);
            var resolved_method = generic_method.Resolve();
            var git = declaringType as GenericInstanceType;
            Collection<TypeReference> genericArguments = git.GenericArguments;
            MethodDefinition definition = new MethodDefinition(method.Name, resolved_method.Attributes, declaringType);

            // EVERYTHING about the method has to be recreated because Mono.Cecil doesn't have any nifty
            // routines to do this for me.

            foreach (var parameter in method.Parameters)
            {
                var ptype = parameter.ParameterType;
                if (ptype.IsGenericParameter)
                {
                    var gp = ptype as GenericParameter;
                    var num = gp.Position;
                    var yo = genericArguments.ToArray()[num];
                    ptype = yo;
                }
                definition.Parameters.Add(new ParameterDefinition(ptype));
            }

            // ImportReference of "reference" returns null. So, instead, copy all instructions over from
            // generic to generic instance method using parameters described in declaringType.
            var body = resolved_method.Body;
            var worker = body.GetILProcessor();
            for (int j = 0; j < body.Instructions.Count; ++j)
            {
                Instruction i = body.Instructions[j];
                object operand = i.Operand;
                Instruction new_inst = i;
                if (operand as GenericParameter != null)
                {
                    var c1 = operand as GenericParameter;
                    var num = c1.Position;
                    var yo = genericArguments.ToArray()[num];
                    new_inst = worker.Create(i.OpCode, yo);
                    new_inst.Offset = i.Offset;
                    body.Instructions.Insert(j, new_inst);
                }
                else if (operand as FieldReference != null)
                {
                    var c1 = operand as FieldReference;
                    var c2 = c1.FieldType;
                    if (c2 != null)
                    {
                        if (c2 as GenericParameter != null)
                        {
                            var c3 = c2 as GenericParameter;
                            var num = c3.Position;
                            var yo = genericArguments.ToArray()[num];
                            // Note, "Name" of the field needs adjusting due to that it's part of a generic instance type.
                            // For now, sluff.
                            string name = c1.Name;
                            var yofields = definition.DeclaringType.Fields;
                            FieldDefinition fd = definition.DeclaringType.Fields.First(f => f.Name == name);
                            c1 = new FieldReference(name, yo);
                            new_inst = worker.Create(i.OpCode, c1);
                        }
                    }
                }

                definition.Body.Instructions.Insert(j, new_inst);
            }
            return definition;
        }
    }
}
