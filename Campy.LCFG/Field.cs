using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mono.Cecil;

namespace Campy.LCFG
{
    public class Field
    {
        public Field(FieldDefinition fieldDefinition, Type declaringType, Type type, int structIndex)
        {
            FieldDefinition = fieldDefinition;
            DeclaringType = declaringType;
            Type = type;
            StructIndex = structIndex;
        }

        /// <summary>
        /// Gets the Cecil field definition.
        /// </summary>
        /// <value>
        /// The Cecil field definition.
        /// </value>
        public FieldDefinition FieldDefinition { get; private set; }

        public Type DeclaringType { get; private set; }

        public Type Type { get; private set; }

        public int StructIndex { get; private set; }
    }
}
