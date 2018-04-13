using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace BoyerMore
{
    /******************************************************************************
     *  Compilation:  javac BoyerMoore.java
     *  Execution:    java BoyerMoore pattern text
     *  Dependencies: StdOut.java
     *
     *  Reads in two strings, the pattern and the input text, and
     *  searches for the pattern in the input text using the
     *  bad-character rule part of the Boyer-Moore algorithm.
     *  (does not implement the strong good suffix rule)
     *
     *  % java BoyerMoore abracadabra abacadabrabracabracadabrabrabracad
     *  text:    abacadabrabracabracadabrabrabracad 
     *  pattern:               abracadabra
     *
     *  % java BoyerMoore rab abacadabrabracabracadabrabrabracad
     *  text:    abacadabrabracabracadabrabrabracad 
     *  pattern:         rab
     *
     *  % java BoyerMoore bcara abacadabrabracabracadabrabrabracad
     *  text:    abacadabrabracabracadabrabrabracad 
     *  pattern:                                   bcara
     *
     *  % java BoyerMoore rabrabracad abacadabrabracabracadabrabrabracad
     *  text:    abacadabrabracabracadabrabrabracad
     *  pattern:                        rabrabracad
     *
     *  % java BoyerMoore abacad abacadabrabracabracadabrabrabracad
     *  text:    abacadabrabracabracadabrabrabracad
     *  pattern: abacad
     *
     ******************************************************************************/

    /**
     *  The {@code BoyerMoore} class finds the first occurrence of a pattern string
     *  in a text string.
     *  <p>
     *  This implementation uses the Boyer-Moore algorithm (with the bad-character
     *  rule, but not the strong good suffix rule).
     *  <p>
     *  For additional documentation,
     *  see <a href="https://algs4.cs.princeton.edu/53substring">Section 5.3</a> of
     *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
     */
    public class BoyerMoore
    {
        private int R;     // the radix
        private int[] right;     // the bad-character skip array

        private char[] pattern;  // store the pattern as a character array
        private string pat;      // or as a string

        /**
         * Preprocesses the pattern string.
         *
         * @param pat the pattern string
         */
        public BoyerMoore(string pat)
        {
            this.R = 256;
            this.pat = pat;

            // position of rightmost occurrence of c in the pattern
            right = new int[R];
            for (int c = 0; c < R; c++)
                right[c] = -1;
            for (int j = 0; j < pat.Length; j++)
                right[pat[j]] = j;
        }

        /**
         * Preprocesses the pattern string.
         *
         * @param pattern the pattern string
         * @param R the alphabet size
         */
        public BoyerMoore(char[] pattern, int R)
        {
            this.R = R;
            this.pattern = new char[pattern.Length];
            for (int j = 0; j < pattern.Length; j++)
                this.pattern[j] = pattern[j];

            // position of rightmost occurrence of c in the pattern
            right = new int[R];
            for (int c = 0; c < R; c++)
                right[c] = -1;
            for (int j = 0; j < pattern.Length; j++)
                right[pattern[j]] = j;
        }

        /**
         * Returns the index of the first occurrrence of the pattern string
         * in the text string.
         *
         * @param  txt the text string
         * @return the index of the first occurrence of the pattern string
         *         in the text string; n if no such match
         */
        public int search(string txt)
        {
            int m = pat.Length;
            int n = txt.Length;
            int skip;
            for (int i = 0; i <= n - m; i += skip)
            {
                skip = 0;
                for (int j = m - 1; j >= 0; j--)
                {
                    if (pat[j] != txt[i+j])
                    {
                        skip = System.Math.Max(1, j - right[txt[i+j]]);
                        break;
                    }
                }
                if (skip == 0) return i;    // found
            }
            return n;                       // not found
        }


        /**
         * Returns the index of the first occurrrence of the pattern string
         * in the text string.
         *
         * @param  text the text string
         * @return the index of the first occurrence of the pattern string
         *         in the text string; n if no such match
         */
        public int search(char[] text)
        {
            int m = pattern.Length;
            int n = text.Length;
            int skip;
            for (int i = 0; i <= n - m; i += skip)
            {
                skip = 0;
                for (int j = m - 1; j >= 0; j--)
                {
                    if (pattern[j] != text[i + j])
                    {
                        skip = System.Math.Max(1, j - right[text[i + j]]);
                        break;
                    }
                }
                if (skip == 0) return i;    // found
            }
            return n;                       // not found
        }


        /**
         * Takes a pattern string and an input string as command-line arguments;
         * searches for the pattern string in the text string; and prints
         * the first occurrence of the pattern string in the text string.
         *
         * @param args the command-line arguments
         */
        public static void main(string[] args)
        {
            string pat = args[0];
            string txt = args[1];
            char[] pattern = pat.ToCharArray();
            char[] text = txt.ToCharArray();

            BoyerMoore boyermoore1 = new BoyerMoore(pat);
            BoyerMoore boyermoore2 = new BoyerMoore(pattern, 256);
            int offset1 = boyermoore1.search(txt);
            int offset2 = boyermoore2.search(text);
        }
    }

    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
        }
    }
}
