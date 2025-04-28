#include <iostream>
#include <vector>
#include <omp.h>
#include <limits.h> // For INT_MAX and INT_MIN
using namespace std;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    int sum = 0;
    double average = 0.0;

    // Parallel Reduction for Sum
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }

    // Parallel Reduction for Min
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val)
            min_val = arr[i];
    }

    // Parallel Reduction for Max
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val)
            max_val = arr[i];
    }

    // Calculate average
    average = (double)sum / n;

    // Output Results
    cout << "\nResults:\n";
    cout << "Sum = " << sum << endl;
    cout << "Minimum = " << min_val << endl;
    cout << "Maximum = " << max_val << endl;
    cout << "Average = " << average << endl;

    return 0;
}
