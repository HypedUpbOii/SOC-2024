#include "../SATSolver/SATsolver.h"
#include <iostream>
using namespace std;
/*

Given a graph G, determine the minimum number of colors needed to be used to color the graph given the constraint that no two adjacent edges
can have the same color.
Graph:
It consists of two objects
1]int numNodes : The number of vertices in the graph
2]vector<pair<int,int>> adjacencyList : The adjacency list of the graph, adjacencyList[i] = {j,k} implies an edge exists between j and k
Constraints:
1]Each vertex must have exactly one color
2]If there exists an edge between vertex i and vertex j, then color(i) != color(j)
To Find : 
minimum color k that satisfies above constraint
(I think) the above problem is NP-Complete, either find a polynomial algo to solve it OR
Use the SAT solver you built as an oracle to solve it

*/
bool isSAT(int i, int numNodes, int numEdges, vector<pair<int,int>>& adjacencyList);

int main() {
    int numNodes;
    cin>>numNodes;
    vector<pair<int,int>> adjacencyList;
    int numEdges;
    cin>>numEdges;
    for (int i=0; i<numEdges; i++) {
        int a,b;
        cin>>a>>b;
        adjacencyList.push_back(make_pair(a,b));
    }
    for (int i = 1; i <= numNodes; i++) {
        if(isSAT(i, numNodes, numEdges, adjacencyList)) {
            cout << i << endl;
            break;
        }
    }
    return 0;
}

bool isSAT(int k, int numNodes, int numEdges, vector<pair<int,int>>& edges) {
    SATSolver s;
    vector<shared_ptr<Formula>> Variables(numNodes * k);
    Variable v;
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < k; j++) {
            v.Name(to_string(i + 1) + "_" + to_string(j + 1));
            Variables[(i * k) + j] = make_shared<Formula>(v);
        }
    }
    for (int i = 0; i < numNodes; i++) {
        auto F = Variables[(i * k)];
        for (int j = 1; j < k; j++) {
            F = Formula::Or(F, Variables[(i * k) + j]);
        }
        s.add(F);
    }
    for (int p = 0; p < numNodes; p++) {
        for (int q = 0; q < k - 1; q++) {
            for (int r = q + 1; r < k; r++) {
                s.add(Formula::Or(Formula::Not(Variables[(p * k) + q]), Formula::Not(Variables[(p * k) + r])));
            }
        }
    }
    for (int i = 0; i < numEdges; i++) {
        for (int j = 0; j < k; j++) {
            int p = edges[i].first - 1;
            int q = edges[i].second - 1;
            s.add(Formula::Or(Formula::Not(Variables[(p * k + j)]), Formula::Not(Variables[(q * k) + j])));
        }
    }
    s.solve();
    return s.result;
}