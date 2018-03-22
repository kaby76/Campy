using System.Collections.Generic;

namespace Campy.Utils
{
    public class Options
    {
        private static Dictionary<string, bool> _boolean_options = new Dictionary<string, bool>();

        public static bool IsOn(string option)
        {
            if (!_boolean_options.ContainsKey(option)) return false;
            var val = _boolean_options[option];
            return val;
        }

        public static void Set(string option, bool value = true)
        {
            _boolean_options[option] = value;
        }
    }
}
