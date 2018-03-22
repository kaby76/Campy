using System.IO;

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
