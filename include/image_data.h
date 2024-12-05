#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#define N 28                // Image size (28x28 pixels)
#define CLASS1_COUNT 100    // Number of "dog" images
#define CLASS2_COUNT 100    // Number of "cat" images
#define TOTAL_SAMPLES (CLASS1_COUNT + CLASS2_COUNT)  // Total number of samples
#define TRAIN_RATIO 0.8     // Ratio for training set size (80%)

// Structure to represent an image sample
typedef struct {
    float data[N * N];  // Flattened image data (28x28 pixels)
    int label;          // Label: 0 for dog, 1 for cat
} Sample;

// Function prototypes related to image data
void normalize_dataset(Sample *dataset, int size);
void split_dataset(Sample *dataset, int total_count, Sample **train_set, Sample **test_set, int *train_count, int *test_count);

#endif // IMAGE_DATA_H
