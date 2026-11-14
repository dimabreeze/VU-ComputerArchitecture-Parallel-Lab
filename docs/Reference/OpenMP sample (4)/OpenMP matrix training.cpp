#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

#include <omp.h>

constexpr int k = 300;
constexpr int n = 300;
constexpr int p = 300;

//define matrixes
int matrix1[n][k] = { 0 };
int matrix2[k][p] = { 0 };
int matrix3[n][p] = { 0 }; 


int main()
{
	// used by OpenMP functions
	int num_threads = 0; 
	int tid = 0;
	int chunk = 4; 

	// time difference between the parallelization
	double diff_time = 0; 

	srand(time(NULL));
	omp_set_num_threads(2);


	// define constants and variables
	int row, col, i, x;
	double start = 0;

	//printin matrix sizes
	printf("Matrix1 size: %dx%d\n", n, k);
	printf("Matrix2 size: %dx%d\n", k, p);
	printf("Matrix3 size: %dx%d\n\n", n, p);


#pragma region Parallel region
#pragma omp parallel shared(matrix1,matrix2,matrix3,num_threads,chunk) private(tid, row, col, i, x)
	{
		x = 10;	//setting the maximum value of matrix A values
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			num_threads = omp_get_num_threads();
			printf("Starting matrix multiple example with %d threads\n", num_threads);
		}

		// Start the timer
		start = omp_get_wtime(); 

#pragma omp for schedule (static, chunk)
		// Matrix1
		for (row = 0; row<n; ++row) {
			for (col = 0; col<k; ++col) {
				//generating random number to matrix1 from 1 to x
				matrix1[row][col] = 1 + (rand() % x);
			}
		}
		


#pragma omp single
		// Matrix2
		for (row = 0; row < k; ++row) {
			for (col = 0; col < p; ++col) {
				//generating randomly 1 or -1 to matrix2
				int temp = rand() % 2;
				if (temp == 0)
					matrix2[row][col] = 1;
				else
					matrix2[row][col] = -1;
			}
		}


#pragma omp for schedule (static, chunk)
		// Matrix multiplication
		for (row = 0; row < n; ++row) {
			for (col = 0; col < p; ++col) {
				int sum = 0;
				// Sum of multiplication products
				for (i = 0; i < k; ++i) {
					sum = sum + matrix1[row][i] * matrix2[i][col];
				}
				matrix3[row][col] = sum; //assigning value to matrix position
			}
		}
	}
	// Parallel region: end
#pragma endregion Parallel region: end

	//print results
	diff_time = omp_get_wtime() - start; //store the difference

	printf("Parallelization took %f sec. time.\n", diff_time);
	printf("******************************************************\n");

	return 0;
}