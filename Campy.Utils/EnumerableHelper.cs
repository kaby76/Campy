using System;
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



        public static IEnumerable<TAccumulate> SelectAggregate<TSource, TAccumulate>(
            this IEnumerable<TSource> source,
            TAccumulate seed,
            Func<TAccumulate, TSource, TAccumulate> func)
        {
            //source.CheckArgumentNull("source");
            //func.CheckArgumentNull("func");
            return source.SelectAggregateIterator(seed, func);
        }

        private static IEnumerable<TAccumulate> SelectAggregateIterator<TSource, TAccumulate>(
            this IEnumerable<TSource> source,
            TAccumulate seed,
            Func<TAccumulate, TSource, TAccumulate> func)
        {
            TAccumulate previous = seed;
            foreach (var item in source)
            {
                TAccumulate result = func(previous, item);
                previous = result;
                yield return result;
            }
        }
    }
}
