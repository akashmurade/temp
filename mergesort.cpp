#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Sequential Bubble Sort
void sequential_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Parallel Bubble Sort using OpenMP
void parallel_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = i % 2; j < n-1; j += 2) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Merge function
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;
    while (i < left.size() && j < right.size()) {
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    }
    while (i < left.size())
        arr[k++] = left[i++];
    while (j < right.size())
        arr[k++] = right[j++];
}

// Sequential Merge Sort
void sequential_merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        sequential_merge_sort(arr, l, m);
        sequential_merge_sort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
void parallel_merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(arr, l, m);
            #pragma omp section
            parallel_merge_sort(arr, m+1, r);
        }
        merge(arr, l, m, r);
    }
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n), temp1, temp2, temp3, temp4;

    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    // Create copies for each sort
    temp1 = arr; temp2 = arr;
    temp3 = arr; temp4 = arr;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    sequential_bubble_sort(temp1);
    end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Time: " << end - start << " seconds\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallel_bubble_sort(temp2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds\n";

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequential_merge_sort(temp3, 0, n-1);
    end = omp_get_wtime();
    cout << "\nSequential Merge Sort Time: " << end - start << " seconds\n";

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallel_merge_sort(temp4, 0, n-1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " seconds\n";

    return 0;
}
