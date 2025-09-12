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


float** generate_matrix_2D_memory(
    int m, // # Rows 
    int n, // # Cols
    int is_random // 0: array of 0.6f, 1: array of random values: [0, 1]
) {
    srand(time(NULL));

    // cache line is 64 bytes 
    float **matrix = (float **)malloc(sizeof(float *) * m); // treat this array as one of int pointers

    // Allocate memory to each row
    for (int i=0; i<m; i++) {
        matrix[i] = malloc(sizeof(float) * n); 
    }

    // Fill the array with float values 
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (is_random) { 
                matrix[i][j] = (float)rand() / RAND_MAX; // random value: [0, 1]
            }
            else {
                matrix[i][j] = 0.0; // init to 0
            }
        }
    }

    return matrix;

}

float* generate_matrix_1D_memory(
    int W, // # Rows
    int H, // # Cols
    int is_random // // 0: array of 0.6f, 1: array of random values: [0, 1]
) {
    float* matrix = malloc(sizeof(float)*W*H); // Define a 1D array -> contiguous memory allocation
 
    // fill with values float32 values
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
    int pW, int pH   // total padding to add to rows and cols
) {
    /* Using multiple threads to apply padding of {0} to pW rows & pH cols of the matrix A in parallel */

    if (aW <= 0 || aH <= 0 || pW < 0 || pH < 0) return NULL;

    int newW = aW + pW; // new number of rows
    int newH = aH + pH; // new number of columns

    int total = newW * newH;

    float* padded_matrix = malloc(total*sizeof(float)); // allocate memory for padded matrix

    for (int i=0; i<total; i++) {
        padded_matrix[i] = 0.0; // init vals to 0
    }

    int p_top = pW/2; // # padded rows before first original row
    int p_left = pH/2; // # padded rows before first original col 


    // Defining a parellised region 
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

void write_results(int num_threads, int kW, int kH, double total_time, int* thread_start_index, int* thread_iterations, float* thread_times) {
    FILE* fptr;
    char filename[256];  // Buffer for filename
    snprintf(filename, sizeof(filename), "performance_metrics_%d_threads.txt", num_threads);
    fptr = fopen(filename, "a");  // Append to file

    // Metrics
    fprintf(fptr, "Assessed Metrics:\n");

    // Print Iterations
    fprintf(fptr, "\nThread Total Iterations distribution:\n");
    for (int t = 0; t < num_threads; t++) {
        fprintf(fptr, "Thread %d: %d Total Iterations\n", t, (thread_iterations[t*64]));
    }


    // Print Thread Starting Points
    fprintf(fptr, "\nThread Ending Locations:\n");
    for (int t = 0; t < num_threads; t++) {
        fprintf(fptr, "Thread %d: Start Location: %d, End Location: %d\n", t, thread_start_index[t*64], thread_start_index[t*64]+thread_iterations[t*64]);
    }

    // Print thread time distribution
    fprintf(fptr, "\nThread time distribution:\n");
    for (int t = 0; t < num_threads; t++) {
        fprintf(fptr, "Thread %d: %f seconds\n", t, thread_times[t*64]);
    }


    // Print Per Thread FLOPS
    double total_ops = 0;
    printf("\nThread FLOPS distribution:\n");
    for (int t = 0; t < num_threads; t++) {
        int thread_ops = thread_iterations[t*64]*2*kW*kH;
        fprintf(fptr, "Thread %d: %f FLOPS\n", t, (thread_ops/thread_times[t*64]));
        total_ops += thread_ops;
    }
    // Print Total FLOPS    
    fprintf(fptr, "Total FLOPS: %f\n", total_ops/total_time);

}

float* flatten_matrix(float** A, int W, int H) {
    /* Convert a matrix of pointers to rows of floats into a contiguous, 1D matrix*/
    float* flat_matrix = malloc(sizeof(float)*H*W);

    for (int i=0; i<W; i++) {
        for (int j=0; j<H; j++) {
            flat_matrix[(i*H)+j] = A[i][j];
        }
    }
    // print_matrix(flat_matrix, W, H);
    return flat_matrix;
}

void get_system_cache_info(long* l1_kb, long* l2_kb, long* l3_kb) {
    /* Retreiving system cache info for utilisation analysis */
    FILE* fp = popen("lscpu | grep -i cache", "r"); // command to identify cache related info
    char line[256];
    
    // Set to (informed) defaults in case parsing fails
    *l1_kb = 32; *l2_kb = 256; *l3_kb = 8192;
    
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "L1d cache:")) {
                sscanf(line, "%*s %*s %ld", l1_kb);
            } else if (strstr(line, "L2 cache:")) {
                float mb; sscanf(line, "%*s %*s %f", &mb);
                *l2_kb = (long)(mb * 1024);
            } else if (strstr(line, "L3 cache:")) {
                float mb; sscanf(line, "%*s %*s %f", &mb);
                *l3_kb = (long)(mb * 1024);
            }
        }
        pclose(fp);
    }
}

double perform_convulution_sequntial(int W, int H, int kW, int kH, int pW, int pH, float* padded_matrix, float* flat_kernel, float* output) {
        /* For benchmarking: calculate the time taken to perform the convulution sequentially (with a single thread)*/
        double start = omp_get_wtime();

        for (int i=0; i<W; i++) { 
            for (int j=0; j<H; j++) {
                
                float score = 0.0; 

                for (int kw=0; kw<kW; kw++) {
                    for (int kh=0; kh<kH; kh++) {
                        score += (padded_matrix[(i+kw)*(H+pH)+j+kh] * flat_kernel[kw*kH+kh]); //2 FLOPS (+&*)
                    }
                }
                output[i*H+j] = score;        

            }
        }

        double end = omp_get_wtime();
        return end-start;
}

double perform_convulution_metrics(int num_threads, int W, int H, int kW, int kH, int pW, int pH, int* thread_iterations, int* thread_start_index, float* padded_matrix, float* flat_kernel, float* output, float* thread_times ) {
   /* Perform the convulution in parallel whilst recording metrics to identify performance & cache-utilisation */

    omp_set_num_threads(num_threads); 

    double start = omp_get_wtime(); // start time of the process

    #pragma omp parallel // parallel region
    {
        int tid = omp_get_thread_num(); // identification of the current thread in the region
        double tid_start = omp_get_wtime(); // start time of the current thread
        int is_first_iteration_j = 1; // flag determining if this is the first iteration for current thread

        #pragma omp for collapse(2) schedule(static) // collapsing the loops into 1, specifying iterations should be scheduled statically among threads
        for (int i=0; i<W; i++) { // iterating through the rows of the matrix
            
            for (int j=0; j<H; j++) { // iterating through the cols of the matrix
                
                float score = 0.0; // to store the 'dot product' with the kernel  

                thread_iterations[tid*64]++; // increment the total iterations for the current thread (each thread is on a new cache line)
                if (is_first_iteration_j) { 
                    thread_start_index[tid*64] = (i*H)+j; // if it is the first iteration, store that index
                    is_first_iteration_j = 0; // reset flag
                    
                }

                for (int kw=0; kw<kW; kw++) { // iterating through rows of the kernel
                    for (int kh=0; kh<kH; kh++) { // iterating through cols in the kernel 
                        score += (padded_matrix[(i+kw)*(H+pH)+j+kh] * flat_kernel[kw*kH+kh]); // computing the dot product: 2 FLOPS (+, *)
                    }
                }
                output[i*H+j] = score; // setting the 'dot product' in the output array
            }
        }
        double tid_end = omp_get_wtime(); // the thread's work is done, store finish time 
        thread_times[tid*64] = tid_end-tid_start; // store the total time the thread spent on its iterations
    }
    double end = omp_get_wtime(); // store the total time for the parallel computations

    return end-start;

}


float perform_convulution_performance(int num_threads, int W, int H, int kW, int kH, int pW, int pH, float* padded_matrix, float* flat_kernel, float* output) {
    /* Assesing Convultion Without the Overhead of Storing Metrics*/

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double tid_start = omp_get_wtime();
        int is_first_iteration_i = 1;
        int is_first_iteration_j = 1;

        #pragma omp for collapse(2) schedule(static)
        for (int i=0; i<W; i++) {
            
            for (int j=0; j<H; j++) {
                
                float score = 0.0; 

                for (int kw=0; kw<kW; kw++) {
                    for (int kh=0; kh<kH; kh++) {
                        score += (padded_matrix[(i+kw)*(H+pH)+j+kh] * flat_kernel[kw*kH+kh]); //2 FLOPS (+&*)
                    }
                }
                output[i*H+j] = score;        
            }
        }
    }
    double end = omp_get_wtime();
    return end-start;

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

    // for memory & parallel efficiency, 'flatten' matricies into contiguous memory
    float* flat_matrix = flatten_matrix(f, W, H);  
    free(f); // no longer required

    float* flat_kernel = flatten_matrix(g, kW, kH);
    free(g);

    // timing the padding process
    double padding_start_time = omp_get_wtime(); 
    float* padded_matrix = apply_padding(flat_matrix, W, H, pW, pH);
    double padding_end_time = omp_get_wtime();

    free(flat_matrix);
    printf("Total Padding Time: %f\n", padding_end_time-padding_start_time);

    /* Calculating Key Metrics */
    int max_threads = omp_get_max_threads();

    // arrays for key metrics - storing each float on diff cacheline
    float* thread_times = calloc(max_threads*64, sizeof(float)); 
    int* thread_iterations = calloc(max_threads*64, sizeof(int));
    int* thread_start_index = calloc(max_threads*64, sizeof(int));
    
    double total_iterations = (double)W*H*kW*kH; // total # of iterations of the convulution algo

    double total_matrix_cache_lines = (double)(W*H*sizeof(float))/64; // total cache lines (that would be) filled by all matrix elements 
    double total_padded_matrix_cache_lines = (double)((W+kW-1)*(H+kH-1)*sizeof(float))/64; // total cache lines filled by all padded-matrix elements 
    double total_kernel_cache_lines = (double)(kW*kH*sizeof(float))/64; // total cache lines filled by all kernel elements 

    // Total cache lines used
    double total_cache_lines = total_padded_matrix_cache_lines + total_kernel_cache_lines;
    double total_cache_lines_per_thread = total_cache_lines / max_threads; 

    double matrix_cache_lines_per_thread = total_matrix_cache_lines/max_threads;
    double padded_matrix_cache_lines_per_thread = total_padded_matrix_cache_lines/max_threads;

    // Get system cache info
    long l1_kb, l2_kb, l3_kb;
    get_system_cache_info(&l1_kb, &l2_kb, &l3_kb);

    // Output Metrics File
    FILE *fptr;
    fptr = fopen("performance_metrics.txt", "a");

    /* Appending Metrics to File */
    fprintf(fptr, "\n=== SYSTEM CACHE INFO ===\n");
    fprintf(fptr, "L1D Cache: %ld KB\n", l1_kb);
    fprintf(fptr, "L2 Cache: %ld KB\n", l2_kb);
    fprintf(fptr, "L3 Cache: %ld KB\n", l3_kb);

    // Calculate if your working set fits
    double working_set_kb = (total_cache_lines * 64) / 1024.0;

    fprintf(fptr, "Working set: %.2f KB\n", working_set_kb);
    fprintf(fptr, "L1 utilization: %.1f%%\n", (working_set_kb / l1_kb) * 100.0);
    fprintf(fptr, "L2 utilization: %.1f%%\n", (working_set_kb / l2_kb) * 100.0);
    fprintf(fptr, "L3 utilization: %.1f%%\n", (working_set_kb / l3_kb) * 100.0);

    fprintf(fptr, "\n=== CACHE LINE ANALYSIS ===\n");
    fprintf(fptr, "Total iterations: %f\n", total_iterations);
    fprintf(fptr, "Total cache lines accessed: %f (%f MB)\n", total_cache_lines, (total_cache_lines * 64) / (1024.0 * 1024.0));
    fprintf(fptr, "Cache lines per thread: %f (%f MB per thread)\n", total_cache_lines_per_thread, (total_cache_lines_per_thread * 64) / (1024.0 * 1024.0));


    fprintf(fptr, "\nTotal Contiguous Matrix Elems: %d\n", W*H);
    fprintf(fptr, "Total Contiguous Kernel Elems: %d\n", kW*kH);
    fprintf(fptr, "Total Contiguous Padded Matrix Elems: %d\n", (W+kW-1)*(H+kH-1));


    fprintf(fptr, "\nTotal Matrix Cache Lines: %f\n", total_matrix_cache_lines);
    fprintf(fptr, "Total Padded Matrix Cache Lines: %f\n", total_padded_matrix_cache_lines);
    fprintf(fptr, "Total Kernel Cache Lines: %f\n", total_kernel_cache_lines);

    fprintf(fptr, "\nTotal Matrix Cache Lines Per Thread (we are not using this): %f\n", matrix_cache_lines_per_thread);
    fprintf(fptr, "Total Padded Matrix Cache Lines Per Thread: %f\n", padded_matrix_cache_lines_per_thread);
    fprintf(fptr, "Total Kernel Cache Lines Per Thread: %f\n", total_kernel_cache_lines/max_threads);
    
    fprintf(fptr, "\nEstimated Cache Line Savings of not Adding Padding: %f\n", 1-(matrix_cache_lines_per_thread/padded_matrix_cache_lines_per_thread));

    printf("Starting Sequential Convultion (performance)\n");
    double sequential_time = perform_convulution_sequntial(W, H, kW, kH, pW, pH, padded_matrix, flat_kernel, output);
    
    printf("Starting Parallel Convultion (performance)\n");
    double parallel_performance_time = perform_convulution_performance(max_threads, W, H, kW, kH, pW, pH, padded_matrix, flat_kernel, output);
    
    printf("Starting Parallel Convultion (metrics)\n");
    double parallel_performance_with_metrics = perform_convulution_metrics(max_threads, W, H, kW, kH, pW, pH, thread_iterations, thread_start_index, padded_matrix, flat_kernel, output, thread_times);

    double parallel_overhead = parallel_performance_time - (sequential_time/max_threads);
    double metric_overhead = parallel_performance_with_metrics - parallel_performance_time;

    fprintf(fptr, "**Results**\n");
    fprintf(fptr, "Sequential Time: %fs\n", sequential_time);
    fprintf(fptr, "Parallel Time (Performance): %f\n", parallel_performance_time);
    fprintf(fptr, "Parallel Time (with metric calculations): %f\n", parallel_overhead);
    fprintf(fptr, "\nParallel Overhead (not considering variance): %f\n", parallel_overhead); // TODO: calculate average overhead
    fprintf(fptr, "Metric Overhead (not considering variance): %f\n", metric_overhead); // TODO: calculate average overhead

    
    free(flat_kernel);
    free(padded_matrix);
    free(thread_times);
    free(thread_start_index);
    free(thread_iterations);
}

int main(int argc, char **argv) {

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