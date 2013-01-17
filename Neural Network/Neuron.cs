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

        public Neuron(double[] inputs)
        {
            SetInputs(inputs);
        }

        /*public Neuron(int inputsSize)
        {
            this.inputs = new double[inputsSize];
            this.weights = new double[inputsSize + 1];
        }*/

        // 3.SETTERS METHODS
        public static void SetBias(double biasToSet)
        {
            bias = biasToSet;
        }

        public static void SetLearningRate(double rate)
        {
            learningRate = rate;
        }

        public void SetInputs(double[] inputs)
        {
            this.inputs = inputs;
            //SortWeights();
        }

        public void SetWeights(double[] weights)
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
         * Sorteia o peso para cada entrada incluindo o bias
         */
        private void SortWeights()
        {
            // Instantiating weights vector
            // First position to store bias weight
            weights = new double[inputs.Length + 1];

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
            // Performing the sum of the inputs with the weights
            double sum = bias * weights[0];
            for ( int i = 0; i < inputs.Length; i++ )
                sum += inputs[ i ] * weights[ i + 1 ];

            // Validation function
            output = Math.Tanh( sum );
        }

        /**
         * Realiza o processo de retro-propagação
         */
        public void Backward( double[] inputs, double expectedOutput )
        {
            SetInputs(inputs);
            SetExpectedOutput(expectedOutput);

            this.Forward();

            // Error calculation
            this.error = this.expectedOutput - this.output;
            
            // Back propagated error calculation
            this.backPropagatedError = (1.0 - this.output * this.output) * this.error;

            // Weights adjustment
            weights[0] += learningRate * bias * backPropagatedError;
            for (int i = 1; i < weights.Length; i++)
                weights[i] += learningRate * this.inputs[i - 1] * backPropagatedError;
        }
    }
}
