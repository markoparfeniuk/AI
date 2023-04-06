using System;
using System.Collections.Generic;

class Queen
{
    public int Row { get; set; }
    public int Column { get; set; }

    public Queen(int row, int column)
    {
        Row = row;
        Column = column;
    }
}

class SimulatedAnnealing
{
    private int n;
    private Queen[] queens;
    private Random rand = new Random();

    public SimulatedAnnealing(int n)
    {
        this.n = n;
        queens = new Queen[n];
    }

    public void Solve()
    {
        double temperature = 1.0;
        double coolingRate = 0.003;

        InitializeQueens();
        PrintBoard();

        while (temperature > 0.1)
        {
            for (int i = 0; i < 100; i++)
            {
                int randomQueen = rand.Next(n);
                int currentPosition = queens[randomQueen].Column;
                int randomPosition = rand.Next(n);

                int currentCost = CalculateCost();
                MoveQueen(randomQueen, randomPosition);
                int newCost = CalculateCost();

                int delta = newCost - currentCost;
                double acceptanceProbability = AcceptanceProbability(delta, temperature);

                if (delta > 0 && rand.NextDouble() > acceptanceProbability)
                {
                    MoveQueen(randomQueen, currentPosition);
                }
            }

            temperature *= 1 - coolingRate;
        }
    }

    private void InitializeQueens()
    {
        for (int i = 0; i < n; i++)
        {
            queens[i] = new Queen(i, rand.Next(n));
        }
    }

    private int CalculateCost()
    {
        int cost = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (queens[i].Row == queens[j].Row ||
                    queens[i].Column == queens[j].Column ||
                    queens[i].Row - queens[j].Row == queens[i].Column - queens[j].Column ||
                    queens[i].Row - queens[j].Row == queens[j].Column - queens[i].Column)
                {
                    cost++;
                }
            }
        }

        return cost;
    }

    private void MoveQueen(int queenIndex, int position)
    {
        queens[queenIndex].Column = position;
    }

    private double AcceptanceProbability(int delta, double temperature)
    {
        if (delta < 0)
        {
            return 1.0;
        }

        return Math.Exp(-delta / temperature);
    }

    public void PrintBoard()
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (queens[i].Column == j)
                {
                    Console.Write("Q ");
                }
                else
                {
                    Console.Write(". ");
                }
            }

            Console.WriteLine();
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        int n = 20;
        SimulatedAnnealing sa = new SimulatedAnnealing(n);
        sa.Solve();
        Console.WriteLine();
        sa.PrintBoard();
    }
}
