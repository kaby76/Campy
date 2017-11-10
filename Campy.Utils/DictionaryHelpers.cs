using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Campy.Utils
{
    public static class DictionaryHelpers
    {
        public static IEnumerable<TValue> PartialMatch<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey partialKey,
            Func<TKey, TKey, bool> comparer)
        {
            // This, or use a RegEx or whatever.
            IEnumerable<TKey> fullMatchingKeys =
                dictionary.Keys.Where(currentKey => comparer(partialKey, currentKey));

            List<TValue> returnedValues = new List<TValue>();

            foreach (TKey currentKey in fullMatchingKeys)
            {
                returnedValues.Add(dictionary[currentKey]);
            }

            return returnedValues;
        }
    }
}
