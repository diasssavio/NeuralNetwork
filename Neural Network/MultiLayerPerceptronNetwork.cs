using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    class MultiLayerPerceptronNetwork
    {
        // ------------------- 1.VARIABLES AND PROPERTIES -------------------
        public double LearningRate { get; set; }

        public double[] Input { get; set; }
        public Neuron[] HiddenLayer { get; set; }
        public Neuron[] OutLayer { get; set; }

        public double[] ExpectedOutput { get; set; }

        //  ------------------- 2.CONSTRUCTORS -------------------
        public MultiLayerPerceptronNetwork(int hiddenLayerAmount, int outLayerAmount, double learningRate)
        {
            LearningRate = learningRate;

            // Allocating Hidden Layer
                HiddenLayer = new Neuron[hiddenLayerAmount];
                // Instantiating Neurons from Hidden Layer
                for (int i = 0; i < HiddenLayer.Length; i++)
                    HiddenLayer[i] = new Neuron(LearningRate);
            //

            // Allocating Output Layer
                OutLayer = new Neuron[outLayerAmount];
                // Instantiating Neurons from Output Layer
                for (int i = 0; i < OutLayer.Length; i++)
                    OutLayer[i] = new Neuron(LearningRate);
            //
            
            // Setting Weights manually
                SetManuallyWeights();
        }

        public MultiLayerPerceptronNetwork(double[] input, int hiddenLayerAmount, int outLayerAmount, double learningRate)
        {
            Input = input;
            LearningRate = learningRate;

            // Allocating Hidden Layer
                HiddenLayer = new Neuron[ hiddenLayerAmount ];
                // Instantiating Neurons from Hidden Layer
                for (int i = 0; i < HiddenLayer.Length; i++)
                    HiddenLayer[i] = new Neuron(LearningRate);
            //

            // Allocating Output Layer
                OutLayer = new Neuron[ outLayerAmount ];
                // Instantiating Neurons from Output Layer
                for (int i = 0; i < OutLayer.Length; i++)
                    OutLayer[i] = new Neuron(LearningRate);
            //

            // Setting Weights manually
            SetManuallyWeights();

            // Set input on the hidden layer
            SetInputOnHiddenLayer();
        }

        //  ------------------- 3.GETTERS & SETTERS -------------------
        /**
         * Retorna o vetor com as saídas de cada neurônio
         */
        public double[] GetOutputs()
        {
            double[] output = new double[ OutLayer.Length ];
            for (int i = 0; i < OutLayer.Length; i++)
                output[i] = OutLayer[i].Output;

            return output;
        }

        /**
         * 
         */
        public double[] GetErrors()
        {
            double[] errors = new double[ HiddenLayer.Length + OutLayer.Length ];
            for (int i = 0; i < OutLayer.Length; i++)
                errors[i] = OutLayer[i].Error;
            for (int i = 0; i < HiddenLayer.Length; i++)
                errors[ i + OutLayer.Length ] = HiddenLayer[i].Error;

            return errors;
        }

        /**
         * 
         */
        public double[] GetBackPropagatedErrors()
        { 
            double[] errors = new double[ HiddenLayer.Length + OutLayer.Length ];
            for (int i = 0; i < OutLayer.Length; i++)
                errors[i] = OutLayer[i].BackPropagatedError;
            for (int i = 0; i < HiddenLayer.Length; i++)
                errors[i + OutLayer.Length] = HiddenLayer[i].BackPropagatedError;

            return errors;
        }

        /**
         * Função para testes da rede que define os pesos manualmente
         */
        private void SetManuallyWeights()
        {
            HiddenLayer[0].Weights = new double[] { -0.2, 0.5, -0.65 };
            HiddenLayer[1].Weights = new double[] { 0.3, 0.72, 0.8 };
            OutLayer[0].Weights = new double[] { -0.5, 1.0, -1.0 };
        }

        /**
         * Define a entrada nos neurônios da camada oculta
         */
        private void SetInputOnHiddenLayer()
        {
            // Put values "input" in the "position" layer
            if (Input != null)
                for (int i = 0; i < HiddenLayer.Length; i++)
                    HiddenLayer[i].Input = Input;
            else throw new Exception("input parameter invalid for function task!");
        }

        /**
         * Define a entrada, input, nos neurônios da camada de saída
         */
        private void SetInputOnOutLayer(double[] input)
        {
            if (input != null)
                for (int i = 0; i < OutLayer.Length; i++)
                    OutLayer[i].Input = input;
            else throw new Exception("input parameter invalid for function task!");
        }

        //  ------------------- 4.FUNCTIONAL METHODS -------------------
        /**
         * Calcula o Erro para cada neurônio na camada oculta
         */
        private void CalculateHiddenLayerErrors()
        {
            for (int i = 0; i < HiddenLayer.Length; i++)
            { 
                // Calculating Error
                double sum = 0.0;
                for(int j = 0; j < OutLayer.Length; j++)
                    sum += OutLayer[j].BackPropagatedError * OutLayer[j].Weights[i + 1];
                HiddenLayer[i].Error = sum;

                // Calculating Back Propagated Error
                HiddenLayer[i].CalculateBackPropagatedError();
            }
        }

        /**
         * Realiza o processo de propagação (forward) na rede
         */
        public void Forward()
        {
            // Do forward process in hidden layer
                // Vector to store the neuron layer outs
                double[] tempOut = new double[ HiddenLayer.Length ];

                // Forward process in current hidden layer and store the outputs
                for (int i = 0; i < HiddenLayer.Length; i++)
                {
                    HiddenLayer[i].Forward();
                    tempOut[i] = HiddenLayer[i].Output;
                }

                // Set inputs from Hidden Layer on Output Layer
                SetInputOnOutLayer(tempOut);
            //

            // Do forward process in the output layer
            for (int i = 0; i < OutLayer.Length; i++)
                OutLayer[i].Forward();
        }

        /**
         * Realiza o processo de retro-propagação (backward) na rede
         */
        public void Backward(double[] inputs, double[] expectedOutput)
        {
            // If expected output vector size equals to output layer size
            if (expectedOutput.Length == OutLayer.Length)
            {
                Input = inputs;
                SetInputOnHiddenLayer();
                ExpectedOutput = expectedOutput;

                // Call to Forward method to realize the respective calculus
                Forward();

                // Calculating Error and Back Propagated Error
                    // Output Layer
                    for (int i = 0; i < OutLayer.Length; i++)
                    {
                        OutLayer[i].ExpectedOutput = expectedOutput[i];
                        OutLayer[i].CalculateError();
                        OutLayer[i].CalculateBackPropagatedError();
                    }

                    // Hidden Layer
                    CalculateHiddenLayerErrors();
                //

                // Weights adjustment
                    // Hidden Layer
                    for (int i = 0; i < HiddenLayer.Length; i++)
                        HiddenLayer[i].WeightsAdjustment();

                    // Output Layer
                    for (int i = 0; i < OutLayer.Length; i++)
                        OutLayer[i].WeightsAdjustment();
                //
            }
            else throw new Exception("Expected Output is in a different size of Output Layer´s Network");
        }
    }
}
