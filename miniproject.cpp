#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <chrono>
using namespace std;

// Partition function for quicksort
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Quicksort function
void quicksort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    vector<int> arr;

    if (rank == 0) {
        cout << "Enter number of elements: ";
        cin >> n;
        arr.resize(n);
        cout << "Enter the elements: ";
        for (int i = 0; i < n; i++) {
            cin >> arr[i];
        }
    }

    // Scatter the data to all processes
    vector<int> localArr(n / size);
    MPI_Scatter(&arr[0], n / size, MPI_INT, &localArr[0], n / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process performs Quicksort on its chunk
    quicksort(localArr, 0, n / size - 1);

    // Gather the sorted chunks back to the root process
    MPI_Gather(&localArr[0], n / size, MPI_INT, &arr[0], n / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Merge all sorted chunks at root (process 0)
    if (rank == 0) {
        // Final sorting after gathering all chunks
        quicksort(arr, 0, n - 1);

        // Measure time for parallel quicksort
        auto start = chrono::high_resolution_clock::now();
        quicksort(arr, 0, n - 1);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> diff = end - start;
        cout << "Parallel Quicksort Time: " << diff.count() << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
