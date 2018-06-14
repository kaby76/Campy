using System;
using System.Collections.Generic;
using System.Text;

namespace Campy.Utils
{
    /// <summary>
    /// LambdaComparer - avoids the need for writing custom IEqualityComparers
    /// 
    /// Usage:
    /// 
    /// List<MyObject> x = myCollection.Except(otherCollection, new LambdaComparer<MyObject>((x, y) => x.Id == y.Id)).ToList();
    /// 
    /// or
    /// 
    /// IEqualityComparer comparer = new LambdaComparer<MyObject>((x, y) => x.Id == y.Id);
    /// List<MyObject> x = myCollection.Except(otherCollection, comparer).ToList();
    /// 
    /// </summary>
    /// <typeparam name="T">The type to compare</typeparam>
    public class LambdaComparer<T> : IEqualityComparer<T>
    {
        private readonly Func<T, T, bool> _lambdaComparer;
        private readonly Func<T, int> _lambdaHash;

        public LambdaComparer(Func<T, T, bool> lambdaComparer) :
            this(lambdaComparer, o => 0)
        {
        }

        public LambdaComparer(Func<T, T, bool> lambdaComparer, Func<T, int> lambdaHash)
        {
            if (lambdaComparer == null)
            {
                throw new ArgumentNullException("lambdaComparer");
            }

            if (lambdaHash == null)
            {
                throw new ArgumentNullException("lambdaHash");
            }

            _lambdaComparer = lambdaComparer;
            _lambdaHash = lambdaHash;
        }

        public bool Equals(T x, T y)
        {
            return _lambdaComparer(x, y);
        }

        public int GetHashCode(T obj)
        {
            return _lambdaHash(obj);
        }
    }
}
