#include "image_data.h"
#include "image_generator.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define EPSILON 0.001f

// function to clear image data (initialize with zeros)
void clear_image(float *image) {
    for (int i = 0; i < N * N; i++) {
        image[i] = 0.0f;
    }
}

// function to check if a point is inside a circle
int is_inside_circle(int x, int y, int cx, int cy, int radius) {
    return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius;
}

// function to generate a "dog-like" image
void generate_dog_image(float *image) {
    clear_image(image);

    int head_center_x = N / 2, head_center_y = N / 2, head_radius = N / 4;
    int ear_radius = N / 8;
    int left_ear_x = N / 4, left_ear_y = N / 4;
    int right_ear_x = 3 * N / 4, right_ear_y = N / 4;
    int nose_radius = N / 16;

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            if (is_inside_circle(x, y, head_center_x, head_center_y, head_radius)) {
                image[x * N + y] = 0.8f;  // head (dim)
            }
            if (is_inside_circle(x, y, left_ear_x, left_ear_y, ear_radius) ||
                is_inside_circle(x, y, right_ear_x, right_ear_y, ear_radius)) {
                image[x * N + y] = 1.0f;  // ears (bright)
            }
            if (is_inside_circle(x, y, head_center_x, head_center_y, nose_radius)) {
                image[x * N + y] = 0.5f;  // nose (dark)
            }
        }
    }
}

// function to check if a point is inside a triangle
int is_inside_triangle(int x, int y, const int v0[2], const int v1[2], const int v2[2]) {
    float area = fabsf(v0[0] * (v1[1] - v2[1]) + v1[0] * (v2[1] - v0[1]) + v2[0] * (v0[1] - v1[1])) / 2.0f;
    float a1 = fabsf(x * (v1[1] - v2[1]) + v1[0] * (v2[1] - y) + v2[0] * (y - v1[1])) / 2.0f;
    float a2 = fabsf(v0[0] * (y - v2[1]) + x * (v2[1] - v0[1]) + v2[0] * (v0[1] - y)) / 2.0f;
    float a3 = fabsf(v0[0] * (v1[1] - y) + v1[0] * (y - v0[1]) + x * (v0[1] - v1[1])) / 2.0f;
    return fabsf(area - (a1 + a2 + a3)) < EPSILON;
}

// function to generate a "cat-like" image
void generate_cat_image(float *image) {
    clear_image(image);

    int head_center_x = N / 2, head_center_y = N / 2, head_radius = N / 4;
    int left_ear[3][2] = {{N / 4, N / 4}, {N / 4, N / 4 + N / 6}, {N / 4 + N / 6, N / 4 + N / 12}};
    int right_ear[3][2] = {{N / 4, 3 * N / 4}, {N / 4, 3 * N / 4 - N / 6}, {N / 4 + N / 6, 3 * N / 4 - N / 12}};
    int nose[3][2] = {{N / 2 + N / 8, N / 2 - N / 16}, {N / 2 + N / 8, N / 2 + N / 16}, {N / 2 + N / 4, N / 2}};

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            if (is_inside_circle(x, y, head_center_x, head_center_y, head_radius)) {
                image[x * N + y] = 0.8f;  // head (dim)
            }
            if (is_inside_triangle(x, y, left_ear[0], left_ear[1], left_ear[2]) ||
                is_inside_triangle(x, y, right_ear[0], right_ear[1], right_ear[2])) {
                image[x * N + y] = 1.0f;  // ears (bright)
            }
            if (is_inside_triangle(x, y, nose[0], nose[1], nose[2])) {
                image[x * N + y] = 0.5f;  // nose (dark)
            }
        }
    }
}

// function to generate all samples
void generate_samples(Sample *samples) {
    for (int i = 0; i < CLASS1_COUNT; i++) {
        generate_dog_image(samples[i].data);
        samples[i].label = 0;  // dog label
    }
    for (int i = 0; i < CLASS2_COUNT; i++) {
        generate_cat_image(samples[CLASS1_COUNT + i].data);
        samples[CLASS1_COUNT + i].label = 1;  // cat label
    }
}

// function to split samples into training and testing sets
void split_samples(Sample *samples, Sample *train_set, Sample *test_set, int *train_count, int *test_count) {
    srand((unsigned int)time(NULL));  // seed the random number generator
    int train_size = (int)(TOTAL_SAMPLES * TRAIN_RATIO);
    int used[TOTAL_SAMPLES] = {0};
    int train_index = 0, test_index = 0;

    while (train_index < train_size) {
        int index = rand() % TOTAL_SAMPLES;
        if (!used[index]) {
            train_set[train_index++] = samples[index];
            used[index] = 1;
        }
    }

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        if (!used[i]) {
            test_set[test_index++] = samples[i];
        }
    }

    *train_count = train_index;
    *test_count = test_index;
}

// function to print an image as a matrix
void print_image(const float *image) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            printf("%.1f ", image[x * N + y]);
        }
        printf("\n");
    }
}

// function to run the entire image generation process
void run_image_generation() {
    Sample samples[TOTAL_SAMPLES];
    Sample train_set[TOTAL_SAMPLES];
    Sample test_set[TOTAL_SAMPLES];
    int train_count = 0, test_count = 0;

    generate_samples(samples);
    split_samples(samples, train_set, test_set, &train_count, &test_count);

    printf("training set size: %d\n", train_count);
    printf("testing set size: %d\n", test_count);

    printf("\nSample dog image from training set:\n");
    for (int i = 0; i < train_count; i++) {
        if (train_set[i].label == 0) {
            print_image(train_set[i].data);
            break;
        }
    }

    printf("\nSample cat image from testing set:\n");
    for (int i = 0; i < test_count; i++) {
        if (test_set[i].label == 1) {
            print_image(test_set[i].data);
            break;
        }
    }
}
