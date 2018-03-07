using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Campy.Utils
{
    public class CampyInfo
    {
        public static string PathOfCampy()
        {
            string full_path = Path.GetDirectoryName(Path.GetFullPath(new CampyInfo().GetType().Assembly.Location))
                                     + Path.DirectorySeparatorChar
                                     ;
            return full_path;
        }
    }
}
