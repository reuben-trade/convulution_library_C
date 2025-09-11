#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include<string.h>

int write_matrix_to_file_text(const char *filename, const float *matrix, int W, int H) {
    if (!filename || !matrix || W <= 0 || H <= 0) return -1;
    FILE *f = fopen(filename, "w");
    if (!f) return -2;
    if (fprintf(f, "%d %d\n", W, H) < 0) {
        fclose(f);
        return -3;
    }
    long total = (long)W * (long)H;
    for (long i = 0; i < total; ++i) {
        if (fprintf(f, "%.9g%c", matrix[i], (i % H == H-1) ? '\n' : ' ') < 0) {
            fclose(f);
            return -4;
        }
    }
    fclose(f);
    return 0;
}

float **read_matrix_from_file_text(const char *filename, int *outW, int *outH) {
    if (!filename || !outW || !outH) {
        printf("DEBUG: NULL parameter passed\n");
        return NULL;
    }

    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("DEBUG: Failed to open file: %s\n", filename);
        perror("fopen");
        return NULL;
    }

    int W, H;
    int scan_result = fscanf(f, "%d %d", &W, &H);
    
    if (scan_result != 2) {
        printf("DEBUG: Failed to read dimensions (expected 2 values, got %d)\n", scan_result);
        fclose(f);
        return NULL;
    }
    if (W <= 0 || H <= 0) { 
        printf("DEBUG: Invalid dimensions W=%d, H=%d\n", W, H);
        fclose(f); 
        return NULL; 
    }

    long total = (long)W * (long)H;
    float *data = (float*) malloc(sizeof(float) * total);
    if (!data) { 
        printf("DEBUG: Failed to allocate memory for data\n");
        fclose(f); 
        return NULL; 
    }

    // Read all float values sequentially
    for (long i = 0; i < total; ++i) {
        if (fscanf(f, "%f", &data[i]) != 1) {
            printf("DEBUG: Failed to read float at position %ld\n", i);
            free(data);
            fclose(f);
            return NULL;
        }
        if (i < 5) { // Print first 5 values for debugging
        }
    }

    // Create array of row pointers
    float **rows = (float**) malloc(sizeof(float*) * W);
    if (!rows) {
        printf("DEBUG: Failed to allocate memory for row pointers\n");
        free(data);
        fclose(f);
        return NULL;
    }
    
    // Each row points into the contiguous data block
    for (int r = 0; r < W; ++r) {
        rows[r] = data + (long)r * H;
    }

    fclose(f);
    *outW = W;
    *outH = H;
    return rows;
}

void free_matrix_2D(float **mat) {
    if (!mat) return;
    free(mat[0]); // free contiguous data block
    free(mat);    // free row pointers
}

void parse_command_line(int argc, char **argv,
                        int *W, int *H, int *kW, int *kH, int *is_random,
                        const char **infile, const char **kinfile, const char **outfile) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-W width] [-H height] [-kW kw] [-kH kh] [-r] [-i infile] [-ki kernelfile] [-o outfile]\n", argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "-W") == 0 && i+1 < argc) {
            *W = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-H") == 0 && i+1 < argc) {
            *H = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-kW") == 0 && i+1 < argc) {
            *kW = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-kH") == 0 && i+1 < argc) {
            *kH = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            *is_random = 1;
        } else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) {
            *infile = argv[++i];
        } else if (strcmp(argv[i], "-ki") == 0 && i+1 < argc) {
            *kinfile = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
            *outfile = argv[++i];
        } else {
            fprintf(stderr, "Unknown or malformed argument: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage.\n");
            exit(1);
        }
    }
}


float** generate_matrix_2D_memory(int m, int n, int is_random) {
    srand(time(NULL));

    // cache line is 64 bytes 
    float **matrix = (float **)malloc(sizeof(float *) * m); // treat this array as one of int pointers
                                                            // Problems:
            
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

float* generate_matrix_1D_memory(int W, int H, int is_random) {
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

    // TODO evaluate
    #pragma omp parallel   
    {    
        #pragma omp for collapse(2) schedule(static)
        for (int i=0; i<aW; ++i) {
            for (int j=0; j<aH; ++j) {
                padded_matrix[(i+p_top)*newH+j+p_left] = A[i*aH+j];
            }
        
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

void conv2d(
    float **f, // input feature map
    int H, // input height,
    int W, // input width
    float **g, // input kernel
    int kH, // kernel height
    int kW, // kernel width
    float *output
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
    

    // float* output_matrix = flatten_matrix(*output, W, H);

    printf("Starting Convultion,\n");
    double start = omp_get_wtime();
    
    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(static)
    for (int i=0; i<W; i++) {
        for (int j=0; j<H; j++) {
            
            float score = 0.0;
            #pragma omp collapse(2) schedule(static) 
            for (int kw=0; kw<kW; kw++) {
                for (int kh=0; kh<kH; kh++) {
                    score += (padded_matrix[(i+kw)*(H+pH)+j+kh] * flat_kernel[kw*kH+kh]);
                }
            }
            output[i*H+j] = score;
    
        }
    }
    }

    double end = omp_get_wtime();
    printf("Final Output in: %fs\n", end-start);

    free(flat_kernel);
    free(padded_matrix);
}

int main(int argc, char **argv) {
    printf("Size of float: %zu bytes\n", sizeof(float));  // Should print 4
    int W = 1024, H = 1024, kW = 90, kH = 90, is_random = 1;
    const char *infile = NULL, *kinfile = NULL, *outfile = NULL;

    parse_command_line(argc, argv, &W, &H, &kW, &kH, &is_random, &infile, &kinfile, &outfile);

    float **f1d = NULL;
    if (infile) {
        int rW, rH;
        f1d = read_matrix_from_file_text(infile, &rW, &rH);
        if (!f1d) { fprintf(stderr, "Failed to read matrix from %s\n", infile); return 1; }
        /* override W,H so dims match the loaded file */
        W = rW; H = rH;
    } else {
        f1d = generate_matrix_2D_memory(W, H, is_random);
        if (!f1d) { fprintf(stderr, "Failed to allocate input matrix\n"); return 1; }
    }

    float **g1d = NULL;
    if (kinfile) {
        int rkW, rkH;
        g1d = read_matrix_from_file_text(kinfile, &rkW, &rkH);
        if (!g1d) { fprintf(stderr, "Failed to read kernel from %s\n", kinfile); free(f1d); return 1; }
        /* override kernel dims */
        kW = rkW; kH = rkH;
    } else {
        g1d = generate_matrix_2D_memory(kW, kH, is_random);
        if (!g1d) { fprintf(stderr, "Failed to allocate kernel matrix\n"); free(f1d); return 1; }
    }

    float *output = generate_matrix_1D_memory(W, H, 0);
    
    /* call your conv2d -- using the parameter order in your posted code */
    conv2d(f1d, H, W, g1d, kH, kW, output);

    /* write output if requested */
    if (outfile) {
        if (write_matrix_to_file_text(outfile, output, W, H) != 0) {
            fprintf(stderr, "Warning: failed to write output to %s\n", outfile);
        }
    }

    free(output);
    return 0;
}