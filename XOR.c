#include<stdlib.h>
#include<stdio.h>
#include<math.h>

//Simple Neural network to learn XOR
double sigmoid(double x){return 1/(1+exp(-x));}
double dsigmoid(double x){return x*(1-x);}
double init_weights(){return ((double)rand())/((double)RAND_MAX);}

void shuffle(int *array,size_t n){
  if(n>1){
    size_t i;
    for(i=0;i<n-1;i++){
      size_t j=i+rand()/(RAND_MAX/(n-i)+1);
      int t=array[j];
      array[j]=array[i];
      array[i]=t;
    }
  }
  
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main(void){
  const double lr=0.1f;

  double hiddenLayer[numHiddenNodes];
  double outputLayer[numOutputs];

  double hiddenLayerBias[numHiddenNodes];
  double outputLayerBias[numOutputs];

  double hiddenWeights[numInputs][numHiddenNodes];
  double outputWeights[numHiddenNodes][numOutputs];

  double training_inputs[numTrainingSets][numInputs]={{0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f}};
  double training_outputs[numTrainingSets][numOutputs]={{0.0f},{1.0f},{1.0f},{0.0f}};

  for(int i=0;i<numInputs;i++){
    for(int j=0;j<numHiddenNodes;j++){
      hiddenWeights[i][j]=init_weights();
    }
  }
  for(int i=0;i<numHiddenNodes;i++){
    for(int j=0;j<numOutputs;j++){
      outputWeights[i][j]=init_weights();
    }
  }

  for(int i=0;i<numOutputs;i++){
      outputLayerBias[i]=init_weights();
  }
  int trainingSetOrder[]={0,1,2,3};

  int numberOfEpochs=10000;

  //Train the neural network for a number of epochs
  for(int epoch=0;epoch<numberOfEpochs;epoch++){
    shuffle(trainingSetOrder,numTrainingSets);

    for(int x=0;x<numTrainingSets;x++){
      int i=trainingSetOrder[x];

      //Forward pass

      //Compute hidden layer activation

      for(int j=0;j<numHiddenNodes;j++){
        double activation = hiddenLayerBias[j];

        for(int k=0;k<numInputs;k++){
          activation+=training_inputs[i][k]*hiddenWeights[k][j];
        }

        hiddenLayer[j] = sigmoid (activation);
        
      }
      //Compute output activation
      for(int j=0;j<numOutputs;j++){
        double activation = outputLayerBias[j];

        for(int k=0;k<numHiddenNodes;k++){
          activation+=hiddenLayer[k]*outputWeights[k][j];
        }

        outputLayer[j] = sigmoid (activation);
      }
      printf("Input: %g %g  Output: %g Predicted Output: %g \n",training_inputs[i][0],training_inputs[i][1],outputLayer[0],training_outputs[i][0]);

      //Backprop

      //Compute change in output weights

      
    }
  }
}
