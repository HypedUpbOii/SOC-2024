#include <iostream>
using namespace std;

/*

TASK : Implement Radix Sort

Binary radix sort is a non-comparative integer sorting algorithm that sorts data by processing individual bits of the numbers.
The algorithm sorts the numbers by examining each bit from the most significant bit (MSB) to the least significant bit (LSB).
For each bit position, it partitions the numbers into two groups: those with the bit set to 0 and those with the bit set to 1.
This process is repeated for each bit position, and the numbers are merged back together after each bit pass. 
Binary radix sort is efficient for fixed-length integer keys, offering a time complexity of O(n) per pass, where n is the number of elements, 
making it suitable for sorting large datasets of integers.
EXPECTED TIME COMPLEXITY : O(n*log(q)) where q = max(arr)

*/

void radixSort (int *arr, int size) {
    int max = 0; // rand() ranges from 0 to 32767
    for (int i = 0; i < size; i++) {
        if (max < arr[i]) max = arr[i];
    }
    for (int exp = 0; (max >> exp) > 0; exp++) {
        int output[size];
        int i, count[2] = {0};
        for (i = 0; i < size; i++) {
            count[(arr[i] >> exp) & 1]++;
        }
        count[1] += count[0];
        for (i = size - 1; i >= 0; i--) {
            output[--count[(arr[i] >> exp) & 1]] = arr[i];
        }
        for (i = 0; i < size; i++) {
            arr[i] = output[i];
        }
    }
}

int main () {

    int n;
    cin>>n;
    int arr[n];
    for (int i=0; i<n; i++) {
        arr[i] = rand();
    }
    radixSort(arr,n);
    for (int i=0; i<n-1; i++) {
        if (arr[i] > arr[i+1]) {
            cout<<"Sorting not done correctly!\n";
            exit(1);
        }
    }
    cout<<"Good job, array is sorted\n";
    return 0;
}