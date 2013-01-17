using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class Neuron
    {
        // 1.ATTRIBUTES
        // Constants
        private static double bias = 1.0;
        private static double learningRate = 0.5;

        // Forward instances
        private double[] inputs;
        private double[] weights; // The bias weight is found in the first position
        private double output;

        // Backward instances
        private double expectedOutput;
        private double error;
        private double backPropagatedError;

        // 2.CONSTRUCTORS
        public Neuron(){}

        public Neuron(double[] input)
        {
            this.Input = input;
        }

        // 3.PROPERTIES
        public static double Bias 
        {
            get { return bias; }
            set { bias = value; }
        }

        public static double LearningRate
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        public double[] Input
        {
            get { return inputs; }
            set { inputs = value; }
        }

        public double[] Weights
        {
            get { return weights; }
            set { weights = value; }
        }

        public double Output
        {
            get { return output; }
            set { output = value; }
        }

        public double ExpectedOutput
        {
            get { return expectedOutput; }
            set { expectedOutput = value; }
        }

        public double Error
        {
            get { return error; }
            set { error = value; }
        }

        public double BackPropagatedError
        {
            get { return backPropagatedError; }
            set { backPropagatedError = value; }
        }

        // 5.FUNCTIONAL METHODS
        /**
         * Sorteia o peso para cada entrada incluindo o bias
         */
        private void SortWeights()
        {
            // Instantiating weights vector
            // First position to store bias weight
            Weights = new double[Input.Length + 1];

            // Instantiating the class to random numbers generation
            Random random = new Random( DateTime.Now.Millisecond );

            // Filling the weights vector
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = random.Next(-1000, 1000) / 1000.0;
        }

        /**
         * Realiza o processo de propagação
         */
        public void Forward()
        {
            // Performing the sum of the inputs with the weights
            double sum = bias * Weights[0];
            for ( int i = 0; i < Input.Length; i++ )
                sum += Input[ i ] * Weights[ i + 1 ];

            // Validation function
            Output = Math.Tanh( sum );
        }

        /**
         * Realiza o processo de retro-propagação
         */
        public void Backward( double[] inputs, double expectedOutput )
        {
            //SetInputs(inputs);
            Input = inputs;
            ExpectedOutput = expectedOutput;
            //SetExpectedOutput(expectedOutput);

            // Call to Forward method to realize the respective calculus
            Forward();

            // Error calculation
            Error = ExpectedOutput - Output;
            
            // Back propagated error calculation
            BackPropagatedError = (1.0 - Output * Output) * Error;

            // Weights adjustment
            Weights[0] += learningRate * bias * BackPropagatedError;
            for (int i = 1; i < Weights.Length; i++)
                Weights[i] += learningRate * Input[i - 1] * BackPropagatedError;
        }
    }
}
