#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

vector<vector<int>> graph;   // dynamic 2D vector
vector<bool> visited;        // dynamic visited array

void parallel_bfs(int start) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int qSize = q.size();

        #pragma omp parallel for
        for (int i = 0; i < qSize; i++) {
            int curr;
            #pragma omp critical
            {
                curr = q.front(); q.pop();
                cout << curr << " ";
            }

            #pragma omp parallel for
            for (int j = 0; j < graph[curr].size(); j++) {
                int adj = graph[curr][j];
                if (!visited[adj]) {
                    #pragma omp critical
                    {
                        visited[adj] = true;
                        q.push(adj);
                    }
                }
            }
        }
    }
}

void parallel_dfs(int start) {
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int curr;
        #pragma omp critical
        {
            curr = s.top(); s.pop();
        }

        if (!visited[curr]) {
            visited[curr] = true;
            cout << curr << " ";

            #pragma omp parallel for
            for (int i = 0; i < graph[curr].size(); i++) {
                int adj = graph[curr][i];
                if (!visited[adj]) {
                    #pragma omp critical
                    {
                        s.push(adj);
                    }
                }
            }
        }
    }
}

int main() {
    int n, m, start;
    cout << "Enter nodes, edges, and start node: ";
    cin >> n >> m >> start;

    graph.resize(n);     // dynamically resize graph for 'n' nodes
    visited.resize(n);   // dynamically resize visited array

    cout << "Enter edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    fill(visited.begin(), visited.end(), false);
    cout << "Parallel BFS: ";
    parallel_bfs(start);

    fill(visited.begin(), visited.end(), false);
    cout << "\nParallel DFS: ";
    parallel_dfs(start);

    return 0;
}
