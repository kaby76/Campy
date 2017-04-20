using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public class Phi : Value
    {
        public Value _v;
        public List<Value> _merge;
        public LLVMCFG.Vertex _block;

        public Phi()
            : base(default(ValueRef))
        { }

        public override string ToString()
        {
            String result = this._v.ToString() + "("
                            + this._merge.Aggregate(
                                new StringBuilder(),
                                (sb, x) =>
                                    sb.Append(x).Append(", "),
                                sb =>
                                {
                                    if (0 < sb.Length)
                                        sb.Length -= 2;
                                    return sb.ToString();
                                })
                            + ")";
            return result;
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
            if ((System.Object) p == null)
            {
                return -1;
            }

            if (p.v < this._v)
                return -1;

            if (p.v > this._v)
                return 1;

            for (int i = 0; i < p.merge.Count; ++i)
            {
                if (i >= this._merge.Count)
                    return -1;
                if (p.merge[i] < this._merge[i])
                    return -1;
                if (p.merge[i] > this._merge[i])
                    return 1;
            }

            return 0;
        }

        public override bool Equals(Object o)
        {
            if (o == null)
            {
                return false;
            }

            if (this.GetType() != o.GetType())
            {
                return false;
            }

            dynamic p = Convert.ChangeType(o, this.GetType());
            if ((System.Object) p == null)
            {
                return false;
            }

            if (p.merge.Count != this._merge.Count)
                return false;

            for (int i = 0; i < p.merge.Count; ++i)
            {
                if (p.merge[i] != this._merge[i])
                    return false;
            }
            return true;
        }

        static public new bool Equals(Object obj1, Object obj2) /* override */
        {
            if (System.Object.ReferenceEquals(obj1, obj2))
                return true;

            if (System.Object.ReferenceEquals(obj1, null))
                return false;

            if (System.Object.ReferenceEquals(obj2, null))
                return false;

            if (obj1.GetType() != obj2.GetType())
                return false;

            dynamic p1 = Convert.ChangeType(obj1, typeof(Phi));
            if ((System.Object) p1 == null)
            {
                return false;
            }

            dynamic p2 = Convert.ChangeType(obj1, typeof(Phi));
            if ((System.Object) p2 == null)
            {
                return false;
            }

            if (p1.merge.Count != p2.merge.Count)
                return false;

            for (int i = 0; i < p1.merge.Count; ++i)
            {
                if (p1.merge[i] != p2.merge[i])
                    return false;
            }
            return true;
        }
    }
}
