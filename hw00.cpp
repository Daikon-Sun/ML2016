#include<iostream>
#include<algorithm>
#include<vector>
#include<fstream>
#include<iomanip>

using namespace std;

double tab[1000][1000];

int main(int argc, char* argv[] ) {
    int qid = atoi(argv[1]);
    ifstream in(argv[2],ios::in);
    ofstream out("ans1.txt",ios::out);
    for(int i = 0; i<500; ++i) for(int j = 0; j<11; ++j)
        in >> tab[i][j];
    vector<double> v;
    for(int i = 0; i<500; ++i) v.push_back( tab[i][qid] );
    sort( v.begin(), v.end() );
    for(int i = 0; i<(int)v.size()-1; ++i)
        out << fixed << setprecision(3) << v[i] << ',';
    out << v.back() << endl;
}
