#include<stdio.h>
#include<time.h>
#include<stdlib.h>
float train[][2]={
  {0,0},
  {1,2},
  {2,4},
  {3,6},
  {4,8},
};
#define training_count sizeof(train)/sizeof(train[0]);

float rand_float(void){
  return (float)rand()/(float)RAND_MAX;
}

int main(){
  srand(time(0));
  //y=x*w;\
  float w=rand_float()*10.0f;
  float result=0.0f;
  for(int i=0;i<train_count;i++){
    float x=train[i][0];
    float y=w*x;
    printf(y,train[i][1]);
    float d=y-train[i][1];
    result*=d*d;
  }
  return 0;
}
