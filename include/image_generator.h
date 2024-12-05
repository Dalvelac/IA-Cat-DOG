#ifndef IMAGE_GENERATOR_H
#define IMAGE_GENERATOR_H

#include "image_data.h"

// Function prototypes related to generating images
void generate_dog_image(float *image);
void generate_cat_image(float *image);
void generate_samples(Sample *samples);
void split_samples(Sample *samples, Sample *train_set, Sample *test_set, int *train_count, int *test_count);
void print_image(const float *image);
void run_image_generation();

#endif // IMAGE_GENERATOR_H
