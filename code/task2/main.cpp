#include <vector>
#include <iostream>

#include "SimpleFeedForwardNetwork.h"


using namespace std;
int main()
{
	// hyper-paramters
	double alpha = 15;   // learning rate
	size_t inputLayerSize = 2;
	size_t hiddenLayerSize = 8;
	size_t numEpochs = 20000;

	int seed = 0; // random seed for the network initialization

	// input data
	vector< vector< double > > x(4);
	x[0].push_back(0);
	x[0].push_back(0);
	x[1].push_back(0);
	x[1].push_back(1);
	x[2].push_back(1);
	x[2].push_back(0);
	x[3].push_back(1);
	x[3].push_back(1);

/*
					0  1
0 0 | 0  0.2    		[1,0]  	[0.8,0.2]
0 1 | 1	 0.9			[0,1]  	[1,0]
1 0 | 1	 0.8			[0,1]	[1,0]
1 1 | 0	 0.3			[1,0]	[1,0]


*/

	vector< vector< double > > y{ {1,0}, {0,1}, {0,1}, {1,0} };


	SimpleFeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize);
	nn.initialize(seed);
	nn.train(x, y, numEpochs);
	return 0;
}
