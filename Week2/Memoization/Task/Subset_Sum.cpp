#include <iostream>
#include <chrono>

#define Loop(i, a, b) for (int i = a; i < b; i++)
#define MOD 1000000007

typedef long long int ll;

using namespace std;
using namespace chrono;
/*
    * THE PROBLEM:
    * Give an array of integers, you need to find the maximum total sum a subset could have such that no two elements
    * in the subset are adjacent.

    * Sample Input:
      6
      6 4 3 2 1 5
    * Sample Output:
      14
    * Explanantion:
    * The subset {6,3,5} which has no two adjacent elements is the one with maximum total sum.
    * TASK:
    * You need to design an O(n) algorithm (ie one which takes a linear time in the input size) to solve this problem.
    * Hint : Think about prefix of the array.
*/
ll Count = 0;
// Declare some variable here for memoization
ll **dp;
ll solution(ll *A, ll n, ll index){
    Count++; // Do not remove this line

    // method 1
    dp[0][0] = 0;
    dp[0][1] = A[0];
    for (ll i = 1; i <= index; i++) {
        dp[i][0] = max(dp[i-1][0], dp[i-1][1]);
        dp[i][1] = (A[i] + dp[i-1][0]) % MOD;
    }
    return max(dp[index][0], dp[index][1]);
    
    // method 2
    // if (index < 0) {
    //     return 0;
    // }
    // if (dp[index][0] != -1) {
    //     return max(dp[index][0], dp[index][1]);
    // }
    // else {
    //     dp[index][0] = solution(A, n, index - 1);
    //     dp[index][1] = (A[index] + (index > 0 ? dp[index - 1][0] : 0)) % MOD;
    // }
    // return max(dp[index][0], dp[index][1]);
}
int main(){
    ll n;
    cin >> n;
    ll *A = new ll[n];
    Loop(i, 0, n) cin >> A[i];
    auto start = high_resolution_clock::now();
    dp = new ll*[n];
    for (ll i = 0; i < n; i++) {
        dp[i] = new ll[2];
        dp[i][0] = -1;
        dp[i][1] = -1;
    }
    ll sum = solution(A, n, n - 1);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<duration<double>>(end - start);
    cout << "Maximum subset sum\t" << sum << "\n";
    cout << "Time spent(s)\t\t" << elapsed.count() << "\n";
    cout << "Number of recursive calls\t" << Count << "\n";
    delete[] A;
}