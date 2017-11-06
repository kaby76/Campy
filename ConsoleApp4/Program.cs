using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Campy;
using System.Numerics;
using Campy.Compiler;
using Campy.Graphs;

namespace ConsoleApp4
{

    class Program
    {
        static void StartDebugging()
        {
            Campy.Utils.Options.Set("graph_trace");
            Campy.Utils.Options.Set("module_trace");
            Campy.Utils.Options.Set("name_trace");
            Campy.Utils.Options.Set("cfg_construction_trace");
            Campy.Utils.Options.Set("dot_graph");
            Campy.Utils.Options.Set("jit_trace");
            Campy.Utils.Options.Set("memory_trace");
            Campy.Utils.Options.Set("ptx_trace");
            Campy.Utils.Options.Set("state_computation_trace");
            Campy.Utils.Options.Set("continue_with_no_resolve");
        }

        bool ApproxEqual(double a, double b)
        {
            if (b > a)
                return (b - a) < 0.01;
            else
                return (a - b) < 0.01;
        }


        static void Main(string[] args)
        {
            Campy.Graphs.Digraph.test();
            Campy.Graphs.DFSPostorder.test();
            string tiny = $@"
13
22
 4  2
 2  3
 3  2
 6  0
 0  1
 2  0
11 12
12  9
 9 10
 9 11
 7  9
10 12
11  4
 4  3
 3  5
 6  8
 8  6
 5  4
 0  5
 6  4
 6  9
 7  6
";

            Digraph G = new Digraph(tiny);
            EdgeClassifier.Classify(G, 7, out Dictionary<DirectedEdge<int>, EdgeClassifier.Classification> result);

            string input2 = @"
14
15
1  5
2  4
2  3
3  9
4  5
5  7
5  6
6  8
7  8
8  2
8  9
9  11
9  10
10 11
12 13
";
            Digraph tt = new Digraph(input2);
            var l = TSortNoBackEdges.Sort(tt, 1);
            var ll = l.ToArray();
            StartDebugging();

            {

                string text = @"The unanimous Declaration of the thirteen United States of America,
                When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.
                We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.
                That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security.
                Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world.
                He has refused his Assent to Laws, the most wholesome and necessary for the public good.
                He has forbidden his Governors to pass Laws of immediate and pressing importance, unless suspended in their operation till his Assent should be obtained; and when so suspended, he has utterly neglected to attend to them.
                He has refused to pass other Laws for the accommodation of large districts of people, unless those people would relinquish the right of Representation in the Legislature, a right inestimable to them and formidable to tyrants only.
                He has called together legislative bodies at places unusual, uncomfortable, and distant from the depository of their Public Records, for the sole purpose of fatiguing them into compliance with his measures.
                He has dissolved Representative Houses repeatedly, for opposing with manly firmness of his invasions on the rights of the people.
                He has refused for a long time, after such dissolutions, to cause others to be elected, whereby the Legislative Powers, incapable of Annihilation, have returned to the People at large for their exercise; the State remaining in the mean time exposed to all the dangers of invasion from without, and convulsions within.
                He has endeavoured to prevent the population of these States; for that purpose obstructing the Laws for Naturalization of Foreigners; refusing to pass others to encourage their migrations hither, and raising the conditions of new Appropriations of Lands.
                He has obstructed the Administration of Justice by refusing his Assent to Laws for establishing Judiciary Powers.
                He has made Judges dependent on his Will alone for the tenure of their offices, and the amount and payment of their salaries.
                He has erected a multitude of New Offices, and sent hither swarms of Officers to harass our people and eat out their substance.
                He has kept among us, in times of peace, Standing Armies without the Consent of our legislatures.
                He has affected to render the Military independent of and superior to the Civil Power.
                He has combined with others to subject us to a jurisdiction foreign to our constitution, and unacknowledged by our laws; giving his Assent to their Acts of pretended Legislation:
                For quartering large bodies of armed troops among us:
                For protecting them, by a mock Trial from punishment for any Murders which they should commit on the Inhabitants of these States:
                For cutting off our Trade with all parts of the world:
                For imposing Taxes on us without our Consent:
                For depriving us in many cases, of the benefit of Trial by Jury:
                For transporting us beyond Seas to be tried for pretended offences:
                For abolishing the free System of English Laws in a neighbouring Province, establishing therein an Arbitrary government, and enlarging its Boundaries so as to render it at once an example and fit instrument for introducing the same absolute rule into these Colonies
                For taking away our Charters, abolishing our most valuable Laws and altering fundamentally the Forms of our Governments:
                For suspending our own Legislatures, and declaring themselves invested with power to legislate for us in all cases whatsoever.
                He has abdicated Government here, by declaring us out of his Protection and waging War against us.
                He has plundered our seas, ravaged our coasts, burnt our towns, and destroyed the lives of our people.
                He is at this time transporting large Armies of foreign Mercenaries to compleat the works of death, desolation, and tyranny, already begun with circumstances of Cruelty & Perfidy scarcely paralleled in the most barbarous ages, and totally unworthy the Head of a civilized nation.
                He has constrained our fellow Citizens taken Captive on the high Seas to bear Arms against their Country, to become the executioners of their friends and Brethren, or to fall themselves by their Hands.
                He has excited domestic insurrections amongst us, and has endeavoured to bring on the inhabitants of our frontiers, the merciless Indian Savages whose known rule of warfare, is an undistinguished destruction of all ages, sexes and conditions.
                In every stage of these Oppressions We have Petitioned for Redress in the most humble terms: Our repeated Petitions have been answered only by repeated injury. A Prince, whose character is thus marked by every act which may define a Tyrant, is unfit to be the ruler of a free people.
                Nor have We been wanting in attentions to our British brethren. We have warned them from time to time of attempts by their legislature to extend an unwarrantable jurisdiction over us. We have reminded them of the circumstances of our emigration and settlement here. We have appealed to their native justice and magnanimity, and we have conjured them by the ties of our common kindred to disavow these usurpations, which, would inevitably interrupt our connections and correspondence. They too have been deaf to the voice of justice and of consanguinity. We must, therefore, acquiesce in the necessity, which denounces our Separation, and hold them, as we hold the rest of mankind, Enemies in War, in Peace Friends.
                We, therefore, the Representatives of the united States of America, in General Congress, Assembled, appealing to the Supreme Judge of the world for the rectitude of our intentions, do, in the Name, and by Authority of the good People of these Colonies, solemnly publish and declare, That these united Colonies are, and of Right ought to be Free and Independent States; that they are Absolved from all Allegiance to the British Crown, and that all political connection between them and the State of Great Britain, is and ought to be totally dissolved; and that as Free and Independent States, they have full Power to levy War, conclude Peace, contract Alliances, establish Commerce, and to do all other Acts and Things which Independent States may of right do. And for the support of this Declaration, with a firm reliance on the protection of divine Providence, we mutually pledge to each other our Lives, our Fortunes and our sacred Honor.
";

                text = text + text + text + text + text + text;
                int n = text.Length;
                string pattern = "He has refused for a long time, after such dissolutions, to cause others to be elected";
                int[] found = new int[n];
                int m = pattern.Length;
                for (int i = 0; i < n; i++)
                {
                    int j;
                    for (j = 0; j < m && i + j < n; j++)
                        if (text[i + j] != pattern[j])
                            break;
                    // mismatch found, break the inner loop
                    if (j == m)
                    {
                        // match found
                        found[i] = 1;
                        found[i+1] = text.IndexOf(pattern, i + 1);
                    }
                }
                for (int i = 0; i < n; ++i)
                    if (found[i] != 0)
                        System.Console.WriteLine(i + " " + found[i]);
                System.Console.WriteLine();

                int[] found2 = new int[n];
                for (int i = 0; i < n - m; i++)
                {
                    found2[i] = text.IndexOf(pattern, i, m) >= 0 ? 1 : 0;
                }
                for (int i = 0; i < n; ++i)
                    if (found[i] != 0)
                        System.Console.WriteLine(i + " " + found[i]);

                Campy.Parallel.For(n, i =>
                {
                    int j;
                    for (j = 0; j < m && i + j < n; j++)
                        if (text[i + j] != pattern[j])
                            break;
                    // mismatch found, break the inner loop
                    if (j == m)
                    {
                        // match found
                        found[i] = 1;
                    }
                });

                Campy.Parallel.For(n, i =>
                {
                    found[i] = text.IndexOf(pattern, i, pattern.Length) >= 0 ? 1 : 0;
                });
            }
        }
    }
}
