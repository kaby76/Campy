using System.Collections.Generic;

namespace Campy.Utils
{
    public static class EnumerableHelper
    {
        /// <summary>
        /// ToIEnumerable converts an IEnumerator into an IEnumerable, so one can use it more easily in
        /// a foreach loop.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="enumerator"></param>
        /// <returns></returns>
        public static IEnumerable<T> ToIEnumerable<T>(this IEnumerator<T> enumerator)
        {
            while (enumerator.MoveNext())
            {
                yield return enumerator.Current;
            }
        }
    }
}
