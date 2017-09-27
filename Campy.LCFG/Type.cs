using System;
using Campy.Types.Utils;
using Swigged.LLVM;

namespace Campy.Compiler
{
    public class Type
    {
        // ECMA 335: There really are many types associated with a "type".
        // Storage type, Underlying type, Reduced type, Verification type,
        // Intermediate type, ..., and these are just a few. See page 34+ for
        // an informative description. Further information on verification types
        // is on page 311. Basically, the stack used to compile
        // must know which type is used for code generation.

        private readonly bool _signed;
        private readonly Mono.Cecil.TypeReference _cil_type;
        private readonly Mono.Cecil.TypeReference _verification_type;
        private readonly Mono.Cecil.TypeReference _stack_verification_type;
        private readonly Mono.Cecil.TypeReference _intermediate_type;
        private readonly TypeRef _intermediate_type_ref;

        public Type(TypeRef intermediate_type, bool signed = true)
        {
            _intermediate_type_ref = intermediate_type;
            _signed = signed;
        }

        public Type(System.Type system_type)
        {
            var mono_type = system_type.ToMonoTypeReference();
            _cil_type = mono_type;
            _verification_type = InitVerificationType(_cil_type);
            _stack_verification_type = InitStackVerificationType(_verification_type);
            _intermediate_type_ref = _stack_verification_type.ToTypeRef();
        }

        public Type(Mono.Cecil.TypeReference mono_type)
        {
            _cil_type = mono_type;
            _verification_type = InitVerificationType(_cil_type);
            _stack_verification_type = InitStackVerificationType(_verification_type);
            _intermediate_type_ref = _stack_verification_type.ToTypeRef();
        }

        private Mono.Cecil.TypeReference InitVerificationType(Mono.Cecil.TypeReference mono_type)
        {
            // Roughly encoding table on page 311.
            if (_cil_type == typeof(sbyte).ToMonoTypeReference())
                return typeof(sbyte).ToMonoTypeReference();
            else if (_cil_type == typeof(byte).ToMonoTypeReference())
                return typeof(sbyte).ToMonoTypeReference();
            else if (_cil_type == typeof(bool).ToMonoTypeReference())
                return typeof(sbyte).ToMonoTypeReference();

            else if (_cil_type == typeof(short).ToMonoTypeReference())
                return typeof(short).ToMonoTypeReference();
            else if (_cil_type == typeof(ushort).ToMonoTypeReference())
                return typeof(short).ToMonoTypeReference();
            else if (_cil_type == typeof(char).ToMonoTypeReference())
                return typeof(short).ToMonoTypeReference();

            else if (_cil_type == typeof(int).ToMonoTypeReference())
                return typeof(int).ToMonoTypeReference();
            else if (_cil_type == typeof(uint).ToMonoTypeReference())
                return typeof(int).ToMonoTypeReference();

            else if (_cil_type == typeof(long).ToMonoTypeReference())
                return typeof(long).ToMonoTypeReference();
            else if (_cil_type == typeof(ulong).ToMonoTypeReference())
                return typeof(long).ToMonoTypeReference();

            else if (_cil_type == typeof(float).ToMonoTypeReference())
                return typeof(float).ToMonoTypeReference();

            else if (_cil_type == typeof(double).ToMonoTypeReference())
                return typeof(double).ToMonoTypeReference();

            else
                return _cil_type;
        }

        private Mono.Cecil.TypeReference InitStackVerificationType(Mono.Cecil.TypeReference mono_type)
        {
            if (_verification_type == typeof(sbyte).ToMonoTypeReference())
                return typeof(int).ToMonoTypeReference();
            else if (_verification_type == typeof(short).ToMonoTypeReference())
                return typeof(int).ToMonoTypeReference();
            else if (_verification_type == typeof(int).ToMonoTypeReference())
                return typeof(int).ToMonoTypeReference();
            else if (_verification_type == typeof(long).ToMonoTypeReference())
                return typeof(long).ToMonoTypeReference();
            else if (_verification_type == typeof(float).ToMonoTypeReference())
                return typeof(float).ToMonoTypeReference();
            else if (_verification_type == typeof(double).ToMonoTypeReference())
                return typeof(double).ToMonoTypeReference();
            else
                return _cil_type;
        }

        public bool is_signed
        {
            get { return _signed; }
        }

        public bool is_unsigned
        {
            get { return !_signed; }
        }

        public Mono.Cecil.TypeReference VerificationType
        {
            get { return _verification_type; }
        }

        public TypeRef IntermediateType
        {
            get { return _intermediate_type_ref; }
        }

        public Mono.Cecil.TypeReference CilType
        {
            get { return _cil_type; }
        }

        public TypeKind GetKind()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind;
        }

        /// Return true if this is one of the six floating-point types
        public bool isFloatingPointTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.HalfTypeKind || kind == TypeKind.FloatTypeKind ||
                   kind == TypeKind.DoubleTypeKind ||
                   kind == TypeKind.X86_FP80TypeKind || kind == TypeKind.FP128TypeKind ||
                   kind == TypeKind.PPC_FP128TypeKind;
        }

        /// Return true if this is 'label'.
        public bool isLabelTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.LabelTypeKind;
        }

        /// Return true if this is 'metadata'.
        public bool isMetadataTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.MetadataTypeKind;
        }

        /// Return true if this is 'token'.
        public bool isTokenTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.TokenTypeKind;
        }

        /// True if this is an instance of IntegerType.
        public bool isIntegerTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.IntegerTypeKind;
        }

        /// Return true if this is an IntegerType of the given width.
        public bool isIntegerTy(uint Bitwidth)
        {
            return false;
        }

        /// Return true if this is an integer type or a vector of integer types.
        public bool isIntOrIntVectorTy()
        {
            return getScalarType().isIntegerTy();
        }

        /// True if this is an instance of FunctionType.
        public bool isFunctionTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.FunctionTypeKind;
        }

        /// True if this is an instance of StructType.
        public bool isStructTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.StructTypeKind;
        }

        /// True if this is an instance of ArrayType.
        public bool isArrayTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.ArrayTypeKind;
        }

        /// True if this is an instance of PointerType.
        public bool isPointerTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.PointerTypeKind;
        }

        /// Return true if this is a pointer type or a vector of pointer types.
        public bool isPtrOrPtrVectorTy()
        {
            return getScalarType().isPointerTy();
        }

        /// True if this is an instance of VectorType.
        public bool isVectorTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            return kind == TypeKind.VectorTypeKind;
        }

        public Type getScalarType()
        {
            return null;
        }

        public UInt32 getPrimitiveSizeInBits()
        {
            TypeKind kind = LLVM.GetTypeKind(_intermediate_type_ref);
            switch (kind)
            {
                case TypeKind.HalfTypeKind: return 16;
                case TypeKind.FloatTypeKind: return 32;
                case TypeKind.DoubleTypeKind: return 64;
                case TypeKind.X86_FP80TypeKind: return 80;
                case TypeKind.FP128TypeKind: return 128;
                case TypeKind.PPC_FP128TypeKind: return 128;
                case TypeKind.X86_MMXTypeKind: return 64;
                case TypeKind.IntegerTypeKind:
                case TypeKind.VectorTypeKind:
                    return LLVM.GetIntTypeWidth(_intermediate_type_ref);
                default: return 0;
            }
        }

        public UInt32 getScalarSizeInBits()
        {
            return getScalarType().getPrimitiveSizeInBits();
        }

        public uint getPointerAddressSpace()
        {
            return LLVM.GetPointerAddressSpace(_intermediate_type_ref);
        }

        public static TypeRef getInt8PtrTy(Swigged.LLVM.ContextRef C, uint AS = 0)
        {
            TypeRef re =  LLVM.Int8TypeInContext(C);
            return re;
        }

        public static TypeRef getIntNTy(ContextRef C, uint AS)
        {
            return LLVM.IntTypeInContext(C, AS);
        }

        public static TypeRef getVoidTy(ContextRef C)
        {
            return LLVM.VoidTypeInContext(C);
        }

        public override bool Equals(object obj)
        {
            if (obj as Type == null) return false;
            return this.IntermediateType.Equals((obj as Type).IntermediateType);
        }

        public override string ToString()
        {
            return _intermediate_type_ref.ToString();
        }

        public static bool operator ==(Type a, Type b)
        {
            if (System.Object.ReferenceEquals(a, b)) return true;
            if (((object) a == null) || ((object) b == null)) return false;
            return a._intermediate_type_ref == b._intermediate_type_ref;
        }

        public static bool operator !=(Type a, Type b)
        {
            return !(a == b);
        }
    }
}
