#ifndef MODEL_H
#define MODEL_H

#include "image_data.h"

#define LEARNING_RATE 0.01
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

void initialize_weights(float *weights, int size);
float forward_pass(float *weights, float *input, int size);
void gradient_descent(Sample *train_set, int train_count, float *weights, int weight_size, int epochs);
void stochastic_gradient_descent(Sample *train_set, int train_count, float *weights, int weight_size, int epochs);
void adam_optimization(Sample *train_set, int train_count, float *weights, int weight_size, int epochs);
void run_training();

#endif // MODEL_H
