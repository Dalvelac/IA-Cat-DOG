#include "image_data.h"
#include <stdlib.h>
#include <math.h>

// function to generate samples
Sample *generate_samples(int num_samples) {
    Sample *samples = (Sample *)malloc(num_samples * sizeof(Sample));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < N * N; j++) {
            samples[i].data[j] = rand() % 256;  // Random pixel value between 0 and 255
        }
        samples[i].label = (i < CLASS1_COUNT) ? 1 : -1;  // Label classes: 1 or -1
    }
    return samples;
}

// function to normalize dataset
void normalize_dataset(Sample *dataset, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < N * N; j++) {
            dataset[i].data[j] /= 255.0;  // Normalize to range 0-1
        }
    }
}

// function to split dataset into training and test sets
void split_dataset(Sample *dataset, int total_samples, Sample **train_set, Sample **test_set, int *train_count, int *test_count) {
    int train_size = (int)(total_samples * TRAIN_RATIO);
    int test_size = total_samples - train_size;

    *train_set = (Sample *)malloc(train_size * sizeof(Sample));
    *test_set = (Sample *)malloc(test_size * sizeof(Sample));

    for (int i = 0; i < train_size; i++) {
        (*train_set)[i] = dataset[i];
    }
    for (int i = 0; i < test_size; i++) {
        (*test_set)[i] = dataset[train_size + i];
    }

    *train_count = train_size;
    *test_count = test_size;
}
