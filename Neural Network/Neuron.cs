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
        private static double bias = 1;
        private static double learningRate = 0.1;

        // Forward instances
        private double[] inputs;
        private double[] weights;
        private double output;

        // Backward instances
        private double expectedOutput;
        private double error;
        private double backPropagatedError;

        // 2.CONSTRUCTORS
        public Neuron(){}

        public Neuron(ref double[] inputs)
        {
            SetInputs(ref inputs);
        }

        // 3.SETTERS METHODS
        public static void SetBias(double biasToSet)
        {
            bias = biasToSet;
        }

        public static void SetLearningRate(double rate)
        {
            learningRate = rate;
        }

        public void SetInputs(ref double[] inputs)
        {
            this.inputs = inputs;
            SortWeights();
        }

        private void SetWeights(ref double[] weights)
        {
            this.weights = weights;
        }

        public void SetExpectedOutput(double expectedOutput)
        {
            this.expectedOutput = expectedOutput;
        }

        // 4.GETTERS METHODS
        public static double GetBias()
        {
            return bias;
        }

        public static double GetLearningRate()
        {
            return learningRate;
        }

        public double[] GetInputs()
        {
            return this.inputs;
        }

        public double GetOutput()
        {
            return this.output;
        }

        public double[] GetWeights()
        {
            return this.weights;
        }

        public double GetExpectedOutput()
        {
            return this.expectedOutput;
        }

        public double GetError()
        {
            return this.error;
        }

        // 5.FUNCTIONAL METHODS
        /**
         * Sorteia o peso para cada entrada
         */
        private void SortWeights()
        {
            // Instantiating weights vector
            weights = new double[inputs.Length + 1]; // Ultima posição para armazenar peso do bias

            // Instantiating the class to random numbers generation
            Random random = new Random( DateTime.Now.Millisecond );

            // Filling the weights vector
            for (int i = 0; i < weights.Length; i++)
                weights[i] = random.Next(-1000, 1000) / 1000.0;
        }

        /**
         * Realiza o processo de propagação
         */
        public void Forward()
        {
            // Performing the sum of the inputsss with the weights
            double sum = 0;
            for ( int i = 0; i < inputs.Length; i++ )
                sum += inputs[ i ] * weights[ i ];
            sum += bias * weights[ weights.Length - 1 ];

            // Validation function
            output = Math.Tanh( sum );
        }

        /**
         * Realiza o processo de retro-propagação
         */
        public void Backward( ref double[] inputs, double expectedOutput )
        {
            SetInputs(ref inputs);
            SetExpectedOutput(expectedOutput);

            // Error calculation
            this.error = this.expectedOutput - this.output;
            
            // Back propagated error calculation
            this.backPropagatedError = (1.0 - this.output * this.output) * this.error;

            // Weights adjustment
            for (int i = 0; i < weights.Length - 1; i++)
                weights[i] += learningRate * this.inputs[i] * backPropagatedError;
            weights[weights.Length - 1] = learningRate * bias * backPropagatedError;
        }
    }
}
