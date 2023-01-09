#include <iostream>
#include <cstdlib>
#include <mpi.h>

using namespace std;

const int N = 1500;

int main(int argc, char** argv) {
    int rank, size;
    double t1, t2;
    int a[N][N], b[N][N], c[N][N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix init
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i * j;
            b[i][j] = i * j;
            c[i][j] = 0;
        }
    }

    int numRows = N / size;
    int startRow = rank * numRows;
    int endRow = startRow + numRows;
    t1 = MPI_Wtime();

    // Matrix multiplication
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    MPI_Allgather(c + startRow, numRows * N, MPI_INT, c, numRows * N, MPI_INT, MPI_COMM_WORLD);

    // Print the result
    if (rank == 0) {
        t2 = MPI_Wtime();
        /*
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cout << c[i][j] << " ";
            }
            cout << endl;
        }
        */
        cout << "Multiplication time: " << t2 - t1 << endl;
    }

    MPI_Finalize();
    return 0;
}