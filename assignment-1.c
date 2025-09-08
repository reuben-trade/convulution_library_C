#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

float** generate_matrix_old(int m, int n, int is_random) {
    srand(time(NULL));

    // cache line is 64 bytes 
    float **matrix = (float **)malloc(sizeof(float *) * m); // treat this array as one of int pointers
                                                            // Problems:
                                                            // m pointers defined represtenting m contiguous blocks of memory
                                                            // malloc sends to heap, so no guarantee rows stored together 
                                                            // 64 / 4 = 16 floats per cache line: potentially redundant memory in cache
    
    for (int i=0; i<m; i++) {
        matrix[i] = malloc(sizeof(float) * n); 
    }

    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (is_random) {
            matrix[i][j] = (float)rand() / RAND_MAX;
            }
            else {
                matrix[i][j] = 0.0;
            }
        }
    }

    return matrix;

}

float* generate_random_matrix(int W, int H, int is_random) {
    float* matrix = malloc(sizeof(float)*W*H);
 
    for (int i=0; i<W*H; i++) {
        if (is_random) {
            matrix[i] = (float)rand()/RAND_MAX;
        }
        else {
            matrix[i] = 0.0;            
        }
    }
    return matrix;
}

float* apply_padding(
    const float *A,   // input matrix (row-major, size aW x aH)
    int aW, int aH,   // input dimensions: rows (aW) and cols (aH)
    int pW, int pH   // total padding to add to rows and cols) 
) {
    // Apply padding of {0} to pW rows & pH cols of the matrix A

    if (aW <= 0 || aH <= 0 || pW < 0 || pH < 0) return NULL;

    int newW = aW + pW; // new number of rows
    int newH = aH + pH; // new number of columns

    int total = newW * newH;

    float* padded_matrix = malloc(total*sizeof(float));


    for (int i=0; i<total; i++) {padded_matrix[i] = 0.0;}

    int p_top = pW/2; // # padded rows before first original row
    int p_left = pH/2; // # padded rows before first original col 

    for (int i=0; i<aW; ++i) {
        for (int j=0; j<aH; ++j) {
            padded_matrix[(i+p_top)*newH+j+p_left] = A[i*aH+j];
        }
        
            
    }
    return padded_matrix;
}


void print_matrix(float* A, int W, int H) {
    for (int i=0; i<W*H; i++) {
        if (i%H == 0) {
            printf("[");

        }

        printf(" %f ", A[i]);

        if ((i+1) % H == 0) {
            printf("]\n");
        }
        
    }
}

float* flatten_matrix(float** A, int W, int H) {
    float* flat_matrix = malloc(sizeof(float)*H*W);

    for (int i=0; i<W; i++) {
        for (int j=0; j<H; j++) {
            flat_matrix[(i*H)+j] = A[i][j];
        }
    }
    // print_matrix(flat_matrix, W, H);
    return flat_matrix;
}

float dot_product(float* A, float* B, int aW, int aH, int bW, int bH) {
    // Apply the dot product to 2 equal matricies, outputting the sum of elements

    float sum = 0.0;
    if ((aW != bW) || (aH != bH)) {
        printf("[ERROR]: Not 2 n*n matricies");
        return 0.0;
    }

    #pragma omp collapse(2) reduction(+:sum)
    for (int i=0; i<aW; i++) {
        for (int j=0; j<aH; j++) {
            sum += A[i*aW+j]*B[i*bW+j]; // potential for reduction & collapse here... possibly depends on kernel size
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
    int pW = kW - 1;
    int pH = kH - 1;

    float* flat_matrix = flatten_matrix(f, W, H);  
    free(f); // no longer required

    float* flat_kernel = flatten_matrix(g, kW, kH);
    free(g);

    float* padded_matrix = apply_padding(flat_matrix, W, H, pW, pH);
    free(flat_matrix);

    // print_matrix(padded_matrix, W+pW, H+pH);
    

    float* output_matrix = generate_random_matrix(W, H, 0);
    float* temp_matrix = generate_random_matrix(kW, kH, 0); 

    printf("Starting Convultion,\n");
    double start = omp_get_wtime();
    
    #pragma omp parallel for //firstprivate(temp_matrix)
    for (int i=0; i<W; i++) {
        for (int j=0; j<H; j++) {
            
            #pragma omp collapse(2)
            for (int kh=0; kh<kH; kh++) {
                for (int kw=0; kw<kW; kw++) {
                    temp_matrix[(kh*kW)+kw] = padded_matrix[((i+kw)*H)+j+kh];
                }
            }
            // #pragma omp task 
            {
            float dp_val = dot_product(temp_matrix, flat_kernel, kW, kH, kW, kH);
            output_matrix[i*H+j] = dp_val;
            }

        }
    }

    double end = omp_get_wtime();
    printf("Final Output in %f:\n", end-start);
    // print_matrix(output_matrix, W, H);
    free(output_matrix);

}

int main() {
    srand(time(NULL));
    int W = 1000;
    int H = 1000;
    int kW = 500;
    int kH = 500;
    
    float** output = malloc(0*sizeof(float *)); // TODO: what is this?

    float** f = generate_matrix_old(W, H, 1);
    float** g = generate_matrix_old(kW, kH, 1);

    conv2d(f, H, W, g, kH, kW, output);
    return 0;
}