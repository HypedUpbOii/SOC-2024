#include "../SATSolver/SATsolver.h"
#include <iostream>
using namespace std;

/*

Given a set V, a list of subsets S and a number k find out if there exists {i_1,i_2,...i_k} where each i_j is distinct 
and |i_j|<|S| and S[i_1] U S[i_2] U S[i_3] ... S[i_k] = V where U represents set union
It is guaranteed all entries of V are unique (i.e. V is not a multiset)
(I think) this problem is NP-Complete
Either solve this in polynomial time OR use the SAT solver

*/
bool isSat(int k, int numVertices, vector<vector<int>>& Subsets);

int main () {
    int lengthOfSet;
    cout<<"Enter length of set : \n";
    cin>>lengthOfSet;
    vector<int> V;
    cout<<"Enter set : \n";
    for (int i=0; i<lengthOfSet; i++) {
        int a;
        cin>>a;
        V.push_back(a);
    }
    int numSubSets;
    cout<<"Enter number of subsets : \n";
    cin>>numSubSets;
    vector<vector<int>> S;
    for (int i=0; i<numSubSets; i++) {
        int lengthOfSubset;
        cout<<"Enter length of subset "<<i<<" : \n";
        cin>>lengthOfSubset;
        cout<<"Enter subset "<<i<<" : \n";
        vector<int> t;
        for (int j=0; j<lengthOfSubset; j++) {
            int x;
            cin>>x;
            t.push_back(x);
        }
        S.push_back(t);
    }

    cout<<"Enter k : \n";
    int k;
    cin>>k;

    if(isSat(k, lengthOfSet, S)) {
        cout << "Possible" << endl;
    } else {
        cout << "Impossible" << endl;
    }
}

bool isSat(int k, int numVertices, vector<vector<int>>& Subsets) {
    SATSolver s;
    vector<shared_ptr<Formula>> Variables(Subsets.size());
    Variable v;
    for (int i = 0; i < Subsets.size(); i++) {
        v.Name(to_string(i + 1));
        Variables[i] = make_shared<Formula>(v);
    }
    vector<vector<int>> varInSubsets(numVertices);
    for (int p = 0; p < numVertices; p++) {
        for (int q = 0; q < Subsets.size(); q++) {
            if (find(Subsets[q].begin(), Subsets[q].end(), p + 1) != Subsets[q].end()) {
                varInSubsets[p].push_back(q);
            }
        }
    }
    for (int i = 0; i < varInSubsets.size(); i++) {
        shared_ptr<Formula> F = Variables[varInSubsets[i][0]];
        for (int j = 1; j < varInSubsets[i].size(); j++) {
            F = Formula::Or(F, Variables[varInSubsets[i][j]]);
        }
        s.add(F);
    }
    vector<vector<shared_ptr<Formula>>> AtMostK(Subsets.size() + 1, vector<shared_ptr<Formula>>(k + 2));
    for (int i = 0; i <= Subsets.size(); i++) {
        v.Name(to_string(i) + "_0");
        AtMostK[i][0] = make_shared<Formula>(v);
        s.add(AtMostK[i][0]);
    }
    for (int j = 1; j <= k + 1; j++) {
        v.Name("0_" + to_string(j));
        AtMostK[0][j] = make_shared<Formula>(v);
        s.add(Formula::Not(AtMostK[0][j]));
    }
    for (int i = 1; i <= Subsets.size(); i++) {
        for (int j = 1; j <= k + 1; j++) {
            v.Name(to_string(i) + "_" + to_string(j));
            AtMostK[i][j] = make_shared<Formula>(v);
            s.add(Formula::Iff(AtMostK[i][j], Formula::Or(Formula::And(Variables[i - 1], AtMostK[i - 1][j - 1]), AtMostK[i - 1][j])));
        }
    }
    s.add(Formula::Not(AtMostK[Subsets.size()][k + 1]));
    s.solve();
    return s.result;
}