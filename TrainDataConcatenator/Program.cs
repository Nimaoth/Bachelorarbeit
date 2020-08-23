using System;
using System.IO;
using System.Linq;

namespace TrainDataConcatenator
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2) {
                Console.Error.WriteLine("more arguments pls");
                return;
            }
            Console.WriteLine($"Output: {args[0]}");

            using var outputFile = new StreamWriter(File.OpenWrite(args[0]));

            outputFile.WriteLine("[");
            bool first = true;
            foreach (var file in args.Skip(1)) {
                if (!first) {
                    outputFile.WriteLine(" ,");
                }
                first = false;

                Console.WriteLine($"Copying {file}");

                var lines = File.ReadAllLines(file);

                if (lines.Length == 1) {
                    outputFile.WriteLine(lines[0].Substring(1, lines[0].Length - 2));
                } else {
                    foreach (var line in lines.Skip(1).SkipLast(1)) {
                        outputFile.WriteLine(line);
                    }
                }
            }
            outputFile.WriteLine("]");

            outputFile.Flush();
        }
    }
}
