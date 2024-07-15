#include <bits/stdc++.h>
using namespace std;

/*

The Naive SAT Solver algorithm is far too slow, thus there is a better algorithm for SAT solving called DPLL
Davis–Putnam–Logemann–Loveland (DPLL) Algorithm:
* First perform unit propogation to get rid of unit clauses (clauses that have only one literal)
* Then, decide the value of some variable and check if the formula becomes SAT (by for example, recursively calling DPLL)
* If it does, return the assignment
* If not, then change the value of that variable 

*/

#define UNDEFINED -1
#define TRUE 1
#define FALSE 0
#define uint unsigned int

class dpll {
    private:
    uint numVariables;
    uint numClauses;
    vector<vector<int>> clauses;
    bool unitProp (vector<int>& partialModel); // top ten bruh moments pass by ref
    bool doPll (vector<int>& partialModel); // same
    
    public:
    vector<int> finalModel;
    void getInput ();
    bool solve();

};



void dpll::getInput () {

    char c = cin.get(); // CNF comments
    while (c == 'c') {
        while (c != '\n') {
            c = cin.get();
        }
        c = cin.get();
    }
    string t;
    cin>>t;

    cin>>numVariables;
    cin>>numClauses;
    for (uint i=0; i<numVariables+1; i++) {
        finalModel.push_back(UNDEFINED);
    }
    for (uint i=0; i<numClauses; i++) {
        vector<int> a;
        int c;
        cin>>c;
        while (c != 0) {
            a.push_back(c);
            cin>>c;
        }
        // a.value = UNDEFINED;
        // a.size = a.elements.size();
        clauses.push_back(a);
    }
}

bool dpll::unitProp (vector<int>& partialModel) {
    while (true) {
        bool unitClause = false;
        for (auto& clause: clauses) {
            uint unassignedVariables = 0;
            int unassignedLiteral = 0;
            bool clauseSatisfied = false;
            for (auto& literal: clause) {
                uint variableNumber = abs(literal);
                int parity = (literal > 0) ? TRUE : FALSE;
                if (partialModel[variableNumber] == UNDEFINED) {
                    unassignedVariables++;
                    unassignedLiteral = literal;
                }
                if (partialModel[variableNumber] == parity) {
                    clauseSatisfied = true;
                    break; // one literal required for clause to be SAT
                }
            }
            if (!clauseSatisfied && (unassignedVariables == 0)) {
                return false; // all literals assigned yet clause is false, contradiction
            }
            if (!clauseSatisfied && (unassignedVariables == 1)) {
                partialModel[abs(unassignedLiteral)] = (unassignedLiteral > 0) ? TRUE : FALSE;
                unitClause = true;
            }
        }
        if (!unitClause) {
            break;
        }
    }
    return true;
}

bool dpll::doPll (vector<int>& partialModel) {
    if (!unitProp(partialModel)) {
        return false;
    }
    bool allClausesSatisfied = true;
    for (auto& clause : clauses) {
        bool clauseSatisfied = false;
        for (auto& literal : clause) {
            uint parity = (literal > 0) ? TRUE : FALSE;
            if (partialModel[abs(literal)] == parity) {
                clauseSatisfied = true;
                break;
            }
        }
        if (!clauseSatisfied) {
            allClausesSatisfied = false;
            break;
        }
    }
    if (allClausesSatisfied) {
        finalModel = partialModel;
        return true;
    }
    for (uint i = 1; i <= numVariables; i++) {
        if (partialModel[i] == UNDEFINED) {
            vector<int> posMod = partialModel;
            posMod[i] = TRUE;
            vector<int> negMod = partialModel;
            negMod[i] = FALSE;
            if (doPll(posMod) || doPll(negMod)) {
                return true;
            }
            return false;
        }
    }
    return false;
}

bool dpll::solve() {
    vector<int> m(numVariables + 1,UNDEFINED); // this one line ruined me
    return doPll(m);
}

int main () {
    
    dpll d;
    d.getInput();

    if (d.solve()) {
        cout<<"SAT\n";
        for (int i=1; i<d.finalModel.size(); i++) {
            cout<<i<<" : "<<d.finalModel[i]<<endl;
        }
    }
    else {
        cout<<"UNSAT\n";
    }
}