using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mono.Cecil;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public class Type
    {

        internal Class Class;
        internal bool IsLocal;

        private TypeRef _type_ref;

        public Type(TypeRef t)
        {
            _type_ref = t;
        }

        public TypeRef T
        {
            get { return _type_ref; }
        }

        /// Return true if this is one of the six floating-point types
        public bool isFloatingPointTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.HalfTypeKind || kind == TypeKind.FloatTypeKind ||
                   kind == TypeKind.DoubleTypeKind ||
                   kind == TypeKind.X86_FP80TypeKind || kind == TypeKind.FP128TypeKind ||
                   kind == TypeKind.PPC_FP128TypeKind;
        }

        /// Return true if this is 'label'.
        public bool isLabelTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.LabelTypeKind;
        }

        /// Return true if this is 'metadata'.
        public bool isMetadataTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.MetadataTypeKind;
        }

        /// Return true if this is 'token'.
        public bool isTokenTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.TokenTypeKind;
        }

        /// True if this is an instance of IntegerType.
        public bool isIntegerTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
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
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.FunctionTypeKind;
        }

        /// True if this is an instance of StructType.
        public bool isStructTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.StructTypeKind;
        }

        /// True if this is an instance of ArrayType.
        public bool isArrayTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.ArrayTypeKind;
        }

        /// True if this is an instance of PointerType.
        public bool isPointerTy()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
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
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
            return kind == TypeKind.VectorTypeKind;
        }

        public Type getScalarType()
        {
            return null;
        }

        public UInt32 getPrimitiveSizeInBits()
        {
            TypeKind kind = LLVM.GetTypeKind(_type_ref);
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
                    return LLVM.GetIntTypeWidth(_type_ref);
                default: return 0;
            }
        }

        public UInt32 getScalarSizeInBits()
        {
            return getScalarType().getPrimitiveSizeInBits();
        }

        public uint getPointerAddressSpace()
        {
            return LLVM.GetPointerAddressSpace(_type_ref);
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

        public Type(TypeReference typeReference, TypeDefinition typeDefinition, TypeRef dataType, TypeRef valueType, TypeRef objectType, StackValueType stackType)
        {
            TypeReferenceCecil = typeReference;
            TypeDefinitionCecil = typeDefinition;
            DataTypeLLVM = dataType;
            ObjectTypeLLVM = objectType;
            StackType = stackType;
            ValueTypeLLVM = valueType;
            DefaultTypeLLVM = stackType == StackValueType.Object ? LLVM.PointerType(ObjectTypeLLVM, 0) : DataTypeLLVM;

            switch (stackType)
            {
                case StackValueType.NativeInt:
                    TypeOnStackLLVM = LLVM.PointerType(LLVM.Int8TypeInContext(LLVM.GetTypeContext(dataType)), 0);
                    break;
                case StackValueType.Float:
                    TypeOnStackLLVM = LLVM.DoubleTypeInContext(LLVM.GetTypeContext(dataType));
                    break;
                case StackValueType.Int32:
                    TypeOnStackLLVM = LLVM.Int32TypeInContext(LLVM.GetTypeContext(dataType));
                    break;
                case StackValueType.Int64:
                    TypeOnStackLLVM = LLVM.Int64TypeInContext(LLVM.GetTypeContext(dataType));
                    break;
                case StackValueType.Value:
                case StackValueType.Object:
                case StackValueType.Reference:
                    TypeOnStackLLVM = DefaultTypeLLVM;
                    break;
            }
        }

        /// <summary>
        /// Gets the LLVM default type.
        /// </summary>
        /// <value>
        /// The LLVM default type.
        /// </value>
        public TypeRef DefaultTypeLLVM { get; private set; }

        /// <summary>
        /// Gets the LLVM object type (object header and <see cref="DataTypeLLVM"/>).
        /// </summary>
        /// <value>
        /// The LLVM boxed type (object header and <see cref="DataTypeLLVM"/>).
        /// </value>
        public TypeRef ObjectTypeLLVM { get; private set; }

        /// <summary>
        /// Gets the LLVM data type.
        /// </summary>
        /// <value>
        /// The LLVM data type (fields).
        /// </value>
        public TypeRef DataTypeLLVM { get; private set; }

        /// <summary>
        /// Gets the LLVM value type.
        /// </summary>
        /// <value>
        /// The LLVM value type (fields).
        /// </value>
        public TypeRef ValueTypeLLVM { get; private set; }

        /// <summary>
        /// Gets the LLVM type when on the stack.
        /// </summary>
        /// <value>
        /// The LLVM type when on the stack.
        /// </value>
        public TypeRef TypeOnStackLLVM { get; private set; }

        /// <summary>
        /// Gets the linkage to use for this type.
        /// </summary>
        /// <value>
        /// The linkage type to use for this type.
        /// </value>
        public Linkage Linkage { get; set; }

        public TypeReference TypeReferenceCecil { get; private set; }
        public TypeDefinition TypeDefinitionCecil { get; private set; }

        public StackValueType StackType { get; private set; }

        public Dictionary<FieldDefinition, Field> Fields { get; set; }

        public TypeState State { get; set; }

        /// <inheritdoc/>
        public override string ToString()
        {
            return TypeReferenceCecil.ToString();
        }
    }
}
