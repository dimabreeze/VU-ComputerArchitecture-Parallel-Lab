// MPI program code
#include"stdio.h"
#include"stdlib.h"
#include"mpi.h"
#include "time.h"

#define N 250
#define X 500
#define Y 120
int main(int argc , char **argv)
{

//define variables

int size,rank,sum=0,i,j,k, darbinis;
int **matrix1;  
int **matrix2; 
int **mat_res; 
double t1,t2,result;
MPI_Status status;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    matrix1 =(int **)malloc(X * sizeof(int*));
    matrix2 =(int **)malloc(N * sizeof(int*));
    mat_res=(int **)malloc(X * sizeof(int*));
 
    for(i = 0; i < X; i++)
    {
        matrix1[i]=(int *)malloc(N * sizeof(int));
        mat_res[i]=(int *)malloc(Y * sizeof(int));
    }
  
    for(i = 0; i < N; i++)
        matrix2[i]=(int *)malloc(Y*sizeof(int));

 //define values of matrix1
srand(time(NULL)); 

    for(i = 0; i < X; i++)
    {
//code

}

//define values of matrix2

    for(i = 0; i < N; i++)
    {
        
//code
    }

MPI_Barrier(MPI_COMM_WORLD);
if(rank == 0)
t1=MPI_Wtime();
    for(i = rank; i < X; i = i+size) 
//divide the task in multiple processes
    {
        for(j = 0; j < Y; j++)
        {
            sum=0;
            for(k = 0; k < N; k++)
            {
                sum = sum + matrix1[i][k] * matrix2[k][j];
            }
            mat_res[i][j] = sum;
        }
    }
 
if(rank != 0) {
    for(i = rank; i < X; i = i+size)
    MPI_Send(&mat_res[i][0], Y, MPI_INT, 0, 10+i, MPI_COMM_WORLD);
}
 
if(rank == 0) {
for(j = 1; j < size; j++)
    {
        for(i = j; i < X; i = i+size)
        {
MPI_Recv(&mat_res[i][0], Y, MPI_INT, j, 10+i, MPI_COMM_WORLD, &status);
        }
    }
}
MPI_Barrier(MPI_COMM_WORLD);
if(rank == 0)
t2 = MPI_Wtime();
result = t2 - t1;

if(rank == 0) {

//print values of matrix1, materix2, mat_res

}

 printf("Time taken = %f seconds\n",result); //time taken
 printf("dydis = #%d \n",size);
 printf("rank = #%d \n",rank); }
 
 MPI_Finalize();
 return 0;
}
