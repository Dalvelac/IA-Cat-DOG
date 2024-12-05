#include "model.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// sgd implementation
void stochastic_gradient_descent(Sample *train_set, int train_count, float *weights, int weight_size, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0;

        // loop through each sample
        for (int i = 0; i < train_count; i++) {
            // perform forward pass to get the output
            float output = forward_pass(weights, train_set[i].data, weight_size);

            // calculate error
            float error = train_set[i].label - output;

            // accumulate loss for monitoring
            loss += error * error;

            // update weights using the computed error (SGD - update after each sample)
            for (int j = 0; j < weight_size - 1; j++) {
                weights[j] += LEARNING_RATE * error * (1 - output * output) * train_set[i].data[j];
            }

            // update bias weight
            weights[weight_size - 1] += LEARNING_RATE * error * (1 - output * output) * 1.0f;  // Bias update
        }

        // print average loss for the epoch
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, loss / train_count);
    }
}

// adam optimization
void adam_optimization(Sample *train_set, int train_count, float *weights, int weight_size, int epochs) {
    // hyperparameters
    float alpha = LEARNING_RATE;  // Learning rate
    float beta1 = BETA1;          // Decay rate for first moment
    float beta2 = BETA2;          // Decay rate for second moment
    float epsilon = EPSILON;      // Small value to avoid division by zero

    // initialize moment estimates and bias corrections
    float m[weight_size];
    float v[weight_size];
    for (int i = 0; i < weight_size; i++) {
        m[i] = 0.0f;
        v[i] = 0.0f;
    }

    // loop over each epoch
    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0;

        // loop through each sample in the training set
        for (int i = 0; i < train_count; i++) {
            // perform forward pass to get the output
            float output = forward_pass(weights, train_set[i].data, weight_size);

            // calculate error
            float error = train_set[i].label - output;

            // accumulate loss for monitoring
            loss += error * error;

            // compute gradient
            float gradient[weight_size];
            for (int j = 0; j < weight_size - 1; j++) {
                gradient[j] = -2 * error * (1 - output * output) * train_set[i].data[j];
            }
            gradient[weight_size - 1] = -2 * error * (1 - output * output) * 1.0f;  // Bias gradient

            // update biased first and second moment estimates
            for (int j = 0; j < weight_size; j++) {
                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];  // Update biased first moment estimate
                v[j] = beta2 * v[j] + (1 - beta2) * (gradient[j] * gradient[j]);  // Update biased second moment estimate
            }

            // correct bias for first and second moments
            float m_hat[weight_size];
            float v_hat[weight_size];
            for (int j = 0; j < weight_size; j++) {
                m_hat[j] = m[j] / (1 - pow(beta1, epoch + 1));
                v_hat[j] = v[j] / (1 - pow(beta2, epoch + 1));
            }

            // update weights
            for (int j = 0; j < weight_size; j++) {
                weights[j] -= alpha * m_hat[j] / (sqrt(v_hat[j]) + epsilon);
            }
        }

        // print average loss for the epoch
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, loss / train_count);
    }
}
