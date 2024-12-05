#include "model.h"
#include "image_data.h"
#include "image_generator.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// function to initialize weights
void initialize_weights(float *weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;  // random weights between -1 and 1
    }
}

// function for forward pass
float forward_pass(const float *weights, const float *input, int size) {
    float sum = 0.0f;

    // calculate dot product of weights and inputs (excluding bias term)
    for (int i = 0; i < size - 1; i++) {
        sum += weights[i] * input[i];
    }

    // add bias term (the last element of weights)
    sum += weights[size - 1] * 1.0f;  // bias is multiplied by 1

    // apply tanh activation function
    return tanhf(sum);
}

// function for gradient descent optimization
void gradient_descent(Sample *train_set, int train_count, float *weights, int weight_size, int epochs) {
    FILE *loss_file = fopen("gd_loss.csv", "w");
    FILE *weights_file = fopen("weights_gd.csv", "w");  // file to save weights

    if (!loss_file || !weights_file) {
        printf("Error opening file for writing.\n");
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0;
        for (int i = 0; i < train_count; i++) {
            // perform forward pass to get the output
            float output = forward_pass(weights, train_set[i].data, weight_size);

            // calculate error
            float error = train_set[i].label - output;

            // accumulate loss for monitoring
            loss += error * error;

            // update weights (gradient descent)
            for (int j = 0; j < weight_size - 1; j++) {
                weights[j] += LEARNING_RATE * error * (1 - output * output) * train_set[i].data[j];
            }

            // update bias weight
            weights[weight_size - 1] += LEARNING_RATE * error * (1 - output * output) * 1.0f;  // bias update
        }

        // print average loss for the epoch
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, loss / train_count);
        fprintf(loss_file, "%d, %.4f\n", epoch + 1, loss / train_count);

        // save weights periodically (e.g., every epoch)
        for (int k = 0; k < weight_size; k++) {
            fprintf(weights_file, "%.4f%s", weights[k], (k == weight_size - 1) ? "\n" : ",");
        }
    }

    fclose(loss_file);
    fclose(weights_file);
}

// new function to run the entire training process for gd, sgd, and adam
void run_training() {
    srand((unsigned int)time(NULL));  // seed for randomness

    // generate dataset
    Sample dataset[TOTAL_SAMPLES];
    generate_samples(dataset);
    normalize_dataset(dataset, TOTAL_SAMPLES);

    // split dataset
    Sample train_set[TOTAL_SAMPLES];
    Sample test_set[TOTAL_SAMPLES];
    int train_count, test_count;
    split_samples(dataset, train_set, test_set, &train_count, &test_count);

    // initialize weights for each algorithm
    int weight_size = N * N + 1;  // number of pixels + 1 bias term

    // timing code for gradient descent
    float weights_gd[weight_size];
    initialize_weights(weights_gd, weight_size);
    printf("\nTraining with gradient descent:\n");

    clock_t start_time = clock();  // start timing GD
    gradient_descent(train_set, train_count, weights_gd, weight_size, 100);
    clock_t end_time = clock();  // end timing GD
    double duration_gd = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Gradient Descent Duration: %.4f seconds\n", duration_gd);

    // training using stochastic gradient descent (sgd)
    float weights_sgd[weight_size];
    initialize_weights(weights_sgd, weight_size);
    printf("\nTraining with stochastic gradient descent (sgd):\n");

    start_time = clock();  // start timing SGD
    stochastic_gradient_descent(train_set, train_count, weights_sgd, weight_size, 100);
    end_time = clock();  // end timing SGD
    double duration_sgd = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Stochastic Gradient Descent Duration: %.4f seconds\n", duration_sgd);

    // training using adam optimization
    float weights_adam[weight_size];
    initialize_weights(weights_adam, weight_size);
    printf("\nTraining with adam optimization:\n");

    start_time = clock();  // start timing Adam
    adam_optimization(train_set, train_count, weights_adam, weight_size, 100);
    end_time = clock();  // end timing Adam
    double duration_adam = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Adam Optimization Duration: %.4f seconds\n", duration_adam);

    // no need to free manually allocated memory since the arrays are locally allocated
}
