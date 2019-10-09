#include "SimpleFeedForwardNetwork.h"
#include <iostream>
#include <random>
#include <iomanip>      // std::setprecision

void SimpleFeedForwardNetwork::initialize(int seed)
{
	srand(seed);
	hiddenLayerWeights.resize(inputLayerSize); //2 -> x1 and x2
	for (size_t i = 0; i < inputLayerSize; i++) {
		hiddenLayerWeights[i].resize(hiddenLayerSize);
		for (size_t j = 0; j < hiddenLayerSize; j++) {
			hiddenLayerWeights[i][j] = (rand() % 100 + 1) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
		}
	}

	outputLayerWeights.resize(2);
	for (size_t i = 0; i < outputLayerWeights.size(); i++) {
		outputLayerWeights[i].resize(hiddenLayerSize);
		for (size_t j = 0; j < hiddenLayerSize; j++){
			outputLayerWeights[i][j] = (rand() % 100 + 1) * 1.0 / 100; 	// This network cannot learn if the initial weights are set to zero.
		}
	}
}

void SimpleFeedForwardNetwork::train(const vector< vector< int > >& x,const vector< vector<int> >& y, size_t numEpochs) {
	size_t trainingexamples = x.size();


	// train the network
	for (size_t epoch = 0; epoch < numEpochs; epoch++) {
		cout << "epoch = " << epoch << endl;
		double loss = 0;
		// for all x training examples
		for (size_t example = 0; example < trainingexamples; example++){
			// propagate the inputs forward to compute the outputs 
			vector< double > activationInput(inputLayerSize); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
				//initisasalise activation of input layer
				activationInput[inputNode] = x[example][inputNode];
			}
			vector< double > activationHidden(hiddenLayerSize);
			// calculate activations of hidden layers (for now, just one hidden layer)
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
				double inputToHidden = 0;
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
					inputToHidden += hiddenLayerWeights[inputNode][hiddenNode] * activationInput[inputNode];
				}
				activationHidden[hiddenNode] = g(inputToHidden);
			}

			// one output node.
			double inputAtOutput1 = 0;
			double inputAtOutput2 = 0;
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
				inputAtOutput1 += outputLayerWeights[0][hiddenNode] * activationHidden[hiddenNode];
				inputAtOutput2 += outputLayerWeights[1][hiddenNode] * activationHidden[hiddenNode];
			}
			double activationOutput1 = g(inputAtOutput1);
			double activationOutput2 = g(inputAtOutput2);
			cout << "Output: [" << std::setprecision(2) << activationOutput1 << ", " << std::setprecision(2) << activationOutput2 << "]";
			cout << "Expected: [" << std::setprecision(2) << y[example][0] << ", " << std::setprecision(2) << y[example][1] << "]";

			// calculating errors
			double errorOfOutputNode1 = gprime(activationOutput1) * (y[example][0] - activationOutput1);
			double errorOfOutputNode2 = gprime(activationOutput2) * (y[example][1] - activationOutput2);
			loss += pow((y[example][0] - activationOutput1),2) + pow((y[example][1] - activationOutput2), 2);
			// double errorOfOutputNodes = (errorOfOutputNode1+errorOfOutputNode2);
			cout << endl;// << "Loss: " << loss << endl;

			// Calculating error of hidden layer. Special calculation since we only have one output node; i.e. no summation over next layer nodes
			// Also adjusting weights of output layer
			vector< double > errorOfHiddenNode(hiddenLayerSize);
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				errorOfHiddenNode[hiddenNode] = outputLayerWeights[0][hiddenNode] * errorOfOutputNode1;
				errorOfHiddenNode[hiddenNode] += outputLayerWeights[1][hiddenNode] * errorOfOutputNode2;
				errorOfHiddenNode[hiddenNode] *= gprime(activationHidden[hiddenNode]);
			}

			//adjusting weights
			//adjusting weights at output layer
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
				outputLayerWeights[0][hiddenNode] += alpha * activationHidden[hiddenNode] * errorOfOutputNode1;
				outputLayerWeights[1][hiddenNode] += alpha * activationHidden[hiddenNode] * errorOfOutputNode2;
			}

			// Adjusting weights at hidden layer.
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
					hiddenLayerWeights[inputNode][hiddenNode] += alpha * activationInput[inputNode] * errorOfHiddenNode[hiddenNode];
				}
			}
			// cout << "Loss: " << loss << endl;
		}
		cout << "Loss: " << loss << endl << endl;
		// cout << endl;
	}

	return;
}
