using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public class Variable : Value
    {
        public String Name;
        static int next;

        public Variable()
            : base(default(ValueRef))
        {
            next++;
            Name = "v" + next;
        }

        public override string ToString()
        {
            return this.Name;
        }

        public override int CompareTo(object o)
        {
            if (o == null)
            {
                return 1;
            }

            if (this.GetType() != o.GetType())
            {
                // Order by string name of type.
                String this_name = this.GetType().FullName;
                String o_name = o.GetType().FullName;
                return String.Compare(this_name, o_name);
            }

            dynamic p = Convert.ChangeType(o, this.GetType());
            if ((System.Object)p == null)
            {
                return -1;
            }

            if (String.Compare(p.Name, this.Name) < 0)
                return -1;

            if (String.Compare(p.Name, this.Name) > 0)
                return 1;

            return 0;
        }

        public override bool Equals(object o)
        {
            if (o == null)
            {
                return false;
            }

            dynamic p = Convert.ChangeType(o, this.GetType());
            if ((System.Object)p == null)
            {
                return false;
            }

            return String.Compare(p.Name, this.Name) == 0;
        }

        public static bool Equals(object obj1, Object obj2) /* override */
        {
            if (System.Object.ReferenceEquals(obj1, obj2))
                return true;

            if (System.Object.ReferenceEquals(obj1, null))
                return false;

            if (System.Object.ReferenceEquals(obj2, null))
                return false;

            if (obj1.GetType() != obj2.GetType())
                return false;

            dynamic p1 = Convert.ChangeType(obj1, typeof(Variable));
            if ((System.Object)p1 == null)
            {
                return false;
            }

            dynamic p2 = Convert.ChangeType(obj1, typeof(Variable));
            if ((System.Object)p2 == null)
            {
                return false;
            }

            return String.Compare(p1.Name, p2.Name) == 0;
        }
    }

}
