#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

float** generate_random_matrix(int W, int H, int is_random) {
    float** matrix = malloc(sizeof(float *)*W);
    for (int i=0; i<W; i++) {
        matrix[i] = malloc(sizeof(float)*H);
    }
    for (int i=0; i<W; i++) {
        for (int j=0; j<H; j++) {
            if (is_random) {
                matrix[i][j] = (float)rand()/RAND_MAX;
            }
            else {
                matrix[i][j] = 0.0;
            }
        }
    }
    return matrix;
}

float** apply_padding(float** A, int aW, int aH, int pW, int pH) {
    // Apply padding of {0} to pW rows & pH cols of the matrix A

    float** padded_matrix = malloc(((aW+pW)*sizeof(float *)));
    for (int i=0; i<aW+pW; i++) {
        padded_matrix[i] = malloc((aH+pH)*sizeof(float));
    }
    int pW_start = pW/2;
    int pH_start = pH/2;

    for (int i=0; i<(aW+pW); i++) {
        for (int j=0; j<(aH+pH); j++) {
            if ((i <= pW_start-1) || (i > aW+pW_start-1)) {
                padded_matrix[i][j] = 0.0;
            }
        
            else if ((j <= pH_start-1) || (j > aH+pH_start-1)) { // remember else if -> otherwise condition is missed
                padded_matrix[i][j] = 0.0;

            }
            else {
                padded_matrix[i][j] = A[i-pW_start][j-pH_start];
            }
            
        }
    }
    return padded_matrix;
}

void print_matrix(float** A, int aW, int aH) {
    for (int i=0; i<aW; i++) {
        for (int j=0; j<aH; j++) {
            if (j%aH == 0) {
                printf("[");
            }
            printf(" %f ", A[i][j]);
            if ((j+1) % aH == 0) {
                printf("]\n");
            }
        }
    }
}

float dot_product(float** A, float** B, int aW, int aH, int bW, int bH) {
    // Apply the dot product to 2 equal matricies, outputting the sum of elements

    float sum = 0.0;
    if ((aW != bW) || (aH != bH)) {
        printf("[ERROR]: Not 2 n*n matricies");
        return 0.0;
    }

    for (int i=0; i<aW; i++) {
        for (int j=0; j<aH; j++) {
            sum += A[i][j]*B[i][j]; // potential for reduction & collapse here... possibly depends on kernel size
        }
    }
    return sum;
}

void conv2d(
    float **f, // input feature map
    int H, // input height,
    int W, // input width
    float **g, // input kernel
    int kH, // kernel height
    int kW, // kernel width
    float **output
) { 
    // we need nW & nH output rows & cols, kernel may restrict output
    // nW % kW = width padding 
    // nH % kH = height padding
    int pW = W % kW;
    int pH = H % kH;

    printf("pW: %d, pH: %d\n", pW, pH);

    float** padded_matrix = apply_padding(f, W, H, pW, pH);
    printf("\nPadded Matrix:\n");
    print_matrix(padded_matrix, W+pW, H+pH);

    free(f); // no longer required

    float** output_matrix = generate_random_matrix(W, H, 0);

    for (int i=0; i<W; i++) {
        for (int j=0; j<H; j++) {

            float** temp_matrix = generate_random_matrix(kW, kH, 0); 
            
            for (int kh=0; kh<kH; kh++) {
                for (int kw=0; kw<kW; kw++) {
                    temp_matrix[kw][kh] = padded_matrix[i+kw][j+kh];
    
                }

            }
            print_matrix(temp_matrix, kW, kH);

            float dp_val = dot_product(temp_matrix, g, kW, kH, kW, kH);
            output_matrix[i][j] = dp_val;

        }
    }
    printf("Final Output:\n");
    print_matrix(output_matrix, W, H);
    free(output_matrix);

}

int main() {
    srand(time(NULL));
    int W = 5;
    int H = 5;
    int kW = 3;
    int kH = 3;

    float** output = malloc(0*sizeof(float *)); // TODO: what is this?

    float** f = generate_random_matrix(W, H, 1);
    float** g = generate_random_matrix(kW, kH, 1);

    print_matrix(f, W, H);
    print_matrix(g, kW, kH);

    conv2d(f, H, W, g, kH, kW, output);
    return 0;
}
