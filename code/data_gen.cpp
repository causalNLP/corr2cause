#include <iostream>
#define ll long long
#include <cstdio>
#include <algorithm>
#include <map>
#include <vector>
#include <string.h>
//#include "json.hpp"
using namespace std;
//using json = nlohmann::json;
const int n = 5;
const int nn = n*(n-1)/2;
const int max_n = 8;
//const int total_graph_number = ;
//const int total_order_number = ;

int power_3[10];
int flag[1000010];
bool f[1000010];
int mp[100][100];
int z[100], degree[100], sum;

//int reconstruct_cache[10010][2];
//std::vector<inlst> reconstruct_cache_shuffle[10010];
//int ci_cache[10010], ci_cache_top;

struct unique_DAG{
    int graph_index_2, graph_index_3, idx;
    std::vector<int> reconstruct_graph, MEC_graph_idx;//reconstruct记录图在order_list的位置， MEC只记录三进制的idx
    std::vector<std::pair<int, int> > graph_edges, CI_relations;
    int pair_relations[max_n][max_n];
    int pair_relation_count[max_n][max_n][7];
} uniqueDag[200010];
int unique_DAG_num = 0;

struct Shuffle_DAG{
    int *order_list;
    int graph_idx;//是下表
}shuffleDag[1000010];
int flag_generated_graph[1000010] = {}, number_shuffleDag = 0;// If the graph is already generated, then skip it.
//using namespace std;
namespace d_seperate{
    inline ll read(){
        char c=getchar();while (c!='-'&&(c<'0'||c>'9'))c=getchar();
        ll k=0,kk=1;if (c=='-')c=getchar(),kk=-1;
        while (c>='0'&&c<='9')k=k*10+c-'0',c=getchar();return kk*k;
    }using namespace std;
    void write(ll x){if (x<0)x=-x,putchar('-');if (x/10)write(x/10);putchar(x%10+'0');}
    void writeln(ll x){write(x);puts("");}
    int edge[110][110];
    int path[10000010][2], path_num;
    int num_hidden, hidden[110];

    int descendent[100];
    int find_descendent(int current_node, int &descendent){
        descendent|=1ll<<(current_node-1);
        for (int i=1;i<=n;i++)
            if (edge[current_node][i] and ((descendent>>i)&1)==0)
                find_descendent(i, descendent);
    }

    int evaluate_condition_set(int condition_nodes, int x, int y){
        if ((condition_nodes>>(x-1))&1 or (condition_nodes>>(y-1))&1)
            return 0;
        // We can not select hidden nodes
        for (int i = 1; i <= num_hidden; i++)
            if ((condition_nodes>>(hidden[i]-1))&1)
                return 0;
        for (int i = 1;i <= path_num; i++) {
            if (condition_nodes&(path[i][0]^path[i][1]))
                continue;
            int flag = 0;
            for (int j = 1; j <= n; j++)
                if (((path[i][1]>>(j-1))&1) && !(condition_nodes&descendent[j])){
                    flag = 1;
                }
            if (!flag) return 0;
        }
        return 1;
    }
    int add_path(int select_nodes, int node_states){
        path_num++;
        path[path_num][0] = select_nodes;
        path[path_num][1] = node_states;
        return 0;
    }
    int find_all_paths(int current, int y, int select_nodes, int node_states, int last_states){
        //last_states: whether current point have input edge
        if (current == y){
            add_path(select_nodes, node_states);
            return 0;
        }
        for (int i = 1;i <= n; i++)
            if (((select_nodes>>(i-1))&1)==0 and (edge[current][i] or edge[i][current])){
                //write(current); putchar(' ');write(i);putchar(' ');writeln(node_states);
                int new_select_nodes = select_nodes|(1ll<<(i-1));
                if (edge[current][i]){
                    find_all_paths(i, y, new_select_nodes, node_states,  1);
                }else{
                    if (last_states == 1){
                        find_all_paths(i, y, new_select_nodes, node_states|(1ll<<(current-1)),  0);
                    }else{
                        find_all_paths(i, y, new_select_nodes, node_states,  0);
                    }
                }
            }
        return 0;
    }
    int set_graph(int graph, unique_DAG &uniqueDag){
        memset(edge, 0, sizeof(edge));
        for (int ii=1;ii<n;ii++)
            for (int jj=ii+1;jj<=n;jj++)
                if ((graph>>mp[ii][jj])&1){
                    edge[ii][jj]=1;
                    uniqueDag.graph_edges.push_back(std::make_pair(ii,jj));
                }
        for (int i = 1; i<=n; i++){
            descendent[i] = 0;
            find_descendent(i, descendent[i]);
            //cout<<i<<' '<<descendent[i]<<endl;
        }
    }
    int calculate_d_seperate(int x, int y, unique_DAG &uniqueDag){
        //cout<<"calc_dsep"<<' '<<uniqueDag.idx<<' '<<x<<' '<<y<<endl;
        path_num = 0;
        num_hidden = 0;
        find_all_paths(x, y, 1<<(x-1), 0, 0);
        /*
        if (uniqueDag.idx==5) {
            puts("Printing all paths");
            writeln(path_num);
            for (int i = 1; i <= path_num; i++){
                write(path[i][0]); putchar(' ');writeln(path[i][1]);
            }
        }*/
        /*if (uniqueDag.idx==2){
            puts("Printing all paths");
            writeln(path_num);
            for (int i = 1; i <= path_num; i++){
                write(path[i][0]); putchar(' ');writeln(path[i][1]);
            }
            //puts("Printing all condition set");
        }*/
        int first_item = (1ll<<(x-1))+(1ll<<(y-1));
        for (int i = 0; i < (1<<n); i++){
            if (evaluate_condition_set(i, x, y)){
                //write(i);putchar(',');
                uniqueDag.CI_relations.push_back(std::make_pair(first_item,i));
                //cout<<"d_sep"<<' '<<uniqueDag.idx<<' '<<first_item<<' '<<i<<endl;
            }
        }
        return 0;
    }

}

namespace reconstruct_graph{

    int search_toposort(int current, int origin_graph){
        if (current == n){
            int new_graph = 0;
            for (int i=1;i<n;i++)
                for (int j=i+1;j<=n;j++)
                    if (z[i]<z[j] and ((origin_graph>>mp[z[i]][z[j]])&1)){
                        //if (origin_graph==1){
                        //    cout<<origin_graph<<' '<<new_graph<<endl;
                        //    cout<<i<<' '<<j<<' '<<z[i]<<' '<<z[j]<<' '<<mp[z[i]][z[j]]<<' '<<mp[i][j]<<endl;
                        //}
                        new_graph+=1ll<<mp[i][j];
                    }
            f[new_graph] = 1;
            //cout<<new_graph<<' '<<origin_graph<<endl;
            //if (origin_graph==1 and new_graph!=1)exit(0);
            //cout<<new_graph<<' '<<origin_graph<<endl;
            return 0;
        }
        for (int i=1;i<=n;i++)
            if (flag[i]==0 and degree[i]==0){
                flag[i]=1;
                for (int j=i+1;j<=n;j++){
                    if ((origin_graph>>mp[i][j])&1){
                        degree[j]-=1;
                    }
                }
                z[current+1] = i;
                search_toposort(current+1, origin_graph);
                flag[i]=0;
                for (int j=i+1;j<=n;j++){
                    if ((origin_graph>>mp[i][j])&1){
                        degree[j]+=1;
                    }
                }
            }
    }
    int wrap_search_toposort(int graph){
        for (int i=1;i<=n;i++)degree[i]=0;
        for (int i=1;i<n;i++)
            for (int j=i+1;j<=n;j++)
                if ((graph>>mp[i][j])&1){
                    degree[j]+=1;
                }
        search_toposort(0, graph);
    }
    int find_all_graph(){
        int idx = 0;
        for (int i=1;i<n;i++)
            for (int j=i+1;j<=n;j++)
                mp[i][j]=idx++;

        int nn = (n-1)*n/2, power_nn = 1ll<<nn;
        //cout<<nn<<' '<<power_nn<<' '<<idx<<endl;
        for (int i=0; i<power_nn; i++)f[i] = 0;
        for (int i=0; i<power_nn; i++){
            //for (int j=0; j<power_nn; j++)cout<<f[j]<<' ';
            //cout<<endl;
            //cout<<"f[i]"<<':'<<(f[i]==0)<<"i:"<<i<<endl;
            //cout<<"begin: "<<i<<endl;
            if (f[i] == 0){
                wrap_search_toposort(i);
                uniqueDag[++unique_DAG_num].graph_index_2 = i;
                uniqueDag[unique_DAG_num].idx = unique_DAG_num;
                d_seperate::set_graph(i, uniqueDag[unique_DAG_num]);
                for (int ii=1;ii<n;ii++)
                    for (int jj=ii+1;jj<=n;jj++){
                        d_seperate::calculate_d_seperate(ii, jj, uniqueDag[unique_DAG_num]);
                    }
                sum+=1;
                //print_dataset(n, i);
                //if(sum==1000){exit(0);}
                //cout<<endl;
            }
            //cout<<"end: "<<i<<endl;
        }
    }
}

namespace node_relations{
    int full_order[1000010][10], num_orders = 0;
    int full_order_stack[10] = {}, flag_full_order[10] = {};
    int generate_full_order(int x){//全排列
        if (x==n){
            num_orders++;
            for (int i=1;i<=n;i++)
                full_order[num_orders][i] = full_order_stack[i];
            return 0;
        }
        for (int i=1; i<=n; i++)
            if (!flag_full_order[i]){
                flag_full_order[i] = 1;
                full_order_stack[x+1]=i;
                generate_full_order(x+1);
                flag_full_order[i] = 0;
            }
    }
    int translate_23(int x){//把二进制图转化为三进制图
        int base = 1, ans = 0;
        //cout<<x<<' ';
        for (;x;x>>=1, base*=3)
            ans+=(x&1)*base;
        //cout<<ans<<endl;
        return ans;
    }

    int reproject_2(int x, int *order_list){//通过一个order_list转换On的点集序列， 用于比较CI
        int ans = 0;
        for (int i=1;i<=n;i++)
            ans+=((x>>(i-1))&1)<<(order_list[i]-1);
        //cout<<"reprojecy: "<<x<<' '<<ans<<' '<<order_list[1]<<' '<<order_list[2]<<' '<<order_list[3]<<endl;
        return ans;
    }

    //我们对order_list的定义是: 如果order[1]=3那么就把原来的1重标号为3
    int reproject_3(int x, int *order_list){//把三进制图按照order_list转换
        int ans = 0;
        for (int i = 1; i<n; i++)
            for (int j = i+1; j<=n; j++){
                int status = x/power_3[mp[i][j]]%3, new_x = order_list[i], new_y = order_list[j];
                if (new_x > new_y and status!=0)status = 3-status, std::swap(new_x, new_y);
                ans += power_3[mp[new_x][new_y]] * status;
                //if(x==1)cout<<i<<' '<<j<<' '<<new_x<<' '<<new_y<<' '<<status<<endl;
            }
        return ans;
    }

    int generate_all_shuffle_graph(unique_DAG& graph){//查询一个图的所有全排列（去重）， 把排列和图对应起来
        //bool flag[1000010] = {};
        //cout<<graph.graph_index_2<<endl;
        //cout<<num_orders<<endl;
        for (int i = 1; i <= num_orders; i++) {
            graph.graph_index_3 = translate_23(graph.graph_index_2);
            int new_graph = reproject_3(graph.graph_index_3, full_order[i]);

            if (!flag_generated_graph[new_graph]){
                flag_generated_graph[new_graph] = ++number_shuffleDag;
                //cout<<full_order[i][1]<<' '<<full_order[i][2]<<' '<<full_order[i][3]<<endl;
                //cout<<graph.graph_index_2<<' '<<new_graph<<endl;
                shuffleDag[number_shuffleDag].graph_idx = graph.idx;
                shuffleDag[number_shuffleDag].order_list = full_order[i];//不确定这里类型对么？
                graph.reconstruct_graph.push_back(new_graph);
            }
        }
    }

    int compare_CI_with_shuffle(unique_DAG& graph_1, unique_DAG& graph_2, int *order_list){
        std::vector<std::pair<int, int> > new_graph_d_sep;
        int len = graph_2.CI_relations.size();

        if (len != graph_1.CI_relations.size())
            return 0;
        for (int i = 0; i < len; i++)
            new_graph_d_sep.push_back(std::make_pair(reproject_2(graph_2.CI_relations[i].first, order_list),
                                                     reproject_2(graph_2.CI_relations[i].second, order_list)));
        std::sort(new_graph_d_sep.begin(), new_graph_d_sep.end());
        for (int i = 0; i < len; i++)
            if ((new_graph_d_sep[i].first != graph_1.CI_relations[i].first)
                || (new_graph_d_sep[i].second != graph_1.CI_relations[i].second))
                return 0;
        return 1;
    }
    int generate_MEC(unique_DAG& graph){//和所有潜在的graph（3^(n^2)）对比，寻找自己的MEC有哪些
        //int len = graph.CI_relations.size();
        //cout<<number_shuffleDag<<endl;
        std::sort(graph.CI_relations.begin(), graph.CI_relations.end());
        for (int i = 1; i <= number_shuffleDag; i++) {
            if (compare_CI_with_shuffle(graph, uniqueDag[shuffleDag[i].graph_idx], shuffleDag[i].order_list)){
                //cout<<graph.idx<<' '<<shuffleDag[i].graph_idx<<' '<<shuffleDag[i].order_list[1]<<' '<<shuffleDag[i].order_list[2]<<' '<<shuffleDag[i].order_list[3]<<endl;
                graph.MEC_graph_idx.push_back(i);
            }
        }
    }


    bool occupied_node[10];
    int node_stack[1000], node_stack_pointer = 0;
    int edges[10][10];
    int compute_relation(unique_DAG& graph, int begin_node, int current_node,  int end_node){
        //用dfs枚举所有可能的路径；返回一个二进制数字
        //[“parent”, “non-parent ancestor”, “child”, “non-child descendant”, “has_collider”, “has_confounder”, “mixed_type”]
        //最暴力版本：枚举所有不重复路径；最后做一波判断
        if (current_node == end_node){
            //cout<<node_stack_pointer<<' '<<begin_node<<' '<<current_node<<' '<<end_node<<endl;
            if (node_stack_pointer<=2){
                //cout<<edges[begin_node][end_node]<<' '<<edges[end_node][begin_node]<<endl;
                if (edges[begin_node][end_node]) return 1;
                if (edges[end_node][begin_node]) return 1<<2;
            }
            int left_length = 0, right_length = 0;
            for (int i = 1; i < node_stack_pointer; i++)
                if (edges[node_stack[i]][node_stack[i+1]]==1) left_length+=1;
                else break;

            for (int i = node_stack_pointer; i > 1; i--)
                if (edges[node_stack[i]][node_stack[i-1]]==1) right_length+=1;
                else break;

            if (left_length == node_stack_pointer-1) return 1<<1;
            if (right_length == node_stack_pointer-1) return 1<<3;
            if (left_length+right_length==node_stack_pointer-1) return 1<<4;
            left_length = 0, right_length = 0;
            for (int i = 1; i < node_stack_pointer; i++)
                if (edges[node_stack[i+1]][node_stack[i]]==1) left_length+=1;
                else break;

            for (int i = node_stack_pointer; i > 1; i++)
                if (edges[node_stack[i-1]][node_stack[i]]==1) right_length+=1;
                else break;

            if (left_length+right_length==node_stack_pointer-1) return 1<<5;
            return 1<<6;
        }
        occupied_node[current_node] = true;
        int relation = 0;
        for (int i = 1;i <= n; i++)
            if (!occupied_node[i] && (edges[current_node][i] || edges[i][current_node])){
                node_stack[++node_stack_pointer] = i;
                relation |= compute_relation(graph, begin_node, i, end_node);
                node_stack[node_stack_pointer--] = 0;
            }
        occupied_node[current_node] = false;
        return relation;
    }

    int count_relation(unique_DAG& graph_1, unique_DAG& graph_2, int* order_list){
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                if (i != j) {
                    int new_x = order_list[i], new_y = order_list[j];
                    for (int k = 0; k < 7; k++) {
                        if ((graph_2.pair_relations[new_x][new_y]>>k)&1)
                            graph_1.pair_relation_count[i][j][k]++;
                    }
                }
    }

    int count_relation_intotal(unique_DAG &graph) {//统计所有MEC里的relation
        int len = graph.MEC_graph_idx.size();
        //cout<<len<<' '<<graph.graph_index_2<<endl;
        for (int i = 0; i < len; i++){
            //cout<<uniqueDag[shuffleDag[graph.MEC_graph_idx[i]].graph_idx].idx<<endl;
            count_relation(graph, uniqueDag[shuffleDag[graph.MEC_graph_idx[i]].graph_idx],
                           shuffleDag[graph.MEC_graph_idx[i]].order_list);
        }
    }

    int generate_node_relations(){
        reconstruct_graph::find_all_graph();
        generate_full_order(0);
        //cout<<"qwq"<<' '<<unique_DAG_num<<endl;
        for (int i = 1; i <= unique_DAG_num; i++)
            generate_all_shuffle_graph(uniqueDag[i]);

        //cout<<"qvq"<<' '<<unique_DAG_num<<endl;
        for (int i = 1; i <= unique_DAG_num; i++)
            generate_MEC(uniqueDag[i]);
        //cout<<"qaq"<<' '<<unique_DAG_num<<endl;
        for (int i = 1; i <= unique_DAG_num; i++) {
            memset(edges, 0, sizeof (edges));
            int len = uniqueDag[i].graph_edges.size();
            for (int j = 0; j < len; j++) {
                edges[uniqueDag[i].graph_edges[j].first][uniqueDag[i].graph_edges[j].second] = 1;
            }
            for (int x = 1; x <= n; x++) {
                for (int y = 1; y <= n; y++)
                    if (x!=y) {
                        node_stack[node_stack_pointer=1] = x;
                        uniqueDag[i].pair_relations[x][y] = compute_relation(uniqueDag[i], x, x, y);
                        //cout<<"pair_relations "<<i<<' '<<x<<' '<<y<<' '<<uniqueDag[i].pair_relations[x][y]<<endl;
                    }
            }
        }
        //cout<<"qnq"<<' '<<unique_DAG_num<<endl;
        for (int i = 1; i <= unique_DAG_num; i++)
            count_relation_intotal(uniqueDag[i]);
        //cout<<"quq"<<' '<<unique_DAG_num<<endl;

    }

}
namespace json_output{

    inline ll read(){
        char c=getchar();while (c!='-'&&(c<'0'||c>'9'))c=getchar();
        ll k=0,kk=1;if (c=='-')c=getchar(),kk=-1;
        while (c>='0'&&c<='9')k=k*10+c-'0',c=getchar();return kk*k;
    }using namespace std;
    void write(ll x){if (x<0)x=-x,putchar('-');if (x/10)write(x/10);putchar(x%10+'0');}
    void writeln(ll x){write(x);puts("");}

    void jump_single_vector(vector<int> &x){
        int len = x.size();
        putchar('[');
        for (int i = 0;i<len;i++){
            cout<<x[i]<<',';
        }putchar(']');putchar(',');
    }
    void jump_pair_vector(vector<pair<int,int> > &x){
        int len = x.size();
        putchar('[');
        for (int i = 0;i<len;i++){
            cout<<'['<<x[i].first<<','<<x[i].second<<']'<<',';
        }putchar(']');putchar(',');
    }
    void jump_single_graph(unique_DAG &graph){
        putchar('{');
        printf("\"graph_index_2\":%d,", graph.graph_index_2);
        printf("\"graph_index_3\":%d,", graph.graph_index_3);
        printf("\"idx\":%d,", graph.idx);
        /*
        "reconstruct_graph":{12344:{1,3,4,2,5}, 12346:[2,3,4,5,1]},
        "MEC_graph":[13456, 13457],
        "graph_edges":[(1, 2), (2, 3)],#2*m
        "CI_relations":[(1, 2), (2,3)],#2*x
        "pair_relations":[],# n*n, 0..2^7-1
	    "pair_relation_count":[] #n*n*7, pair_relations in all MEC
        std::vector<int> reconstruct_graph, MEC_graph_idx;//reconstruct记录图在order_list的位置， MEC只记录三进制的idx
        std::vector<std::pair<int, int>> graph_edges, CI_relations;
        int pair_relations[max_n][max_n];
        int pair_relation_count[max_n][max_n][7];
         */

        printf("\"graph_edges\":");jump_pair_vector(graph.graph_edges);
        printf("\"CI_relations\":");jump_pair_vector(graph.CI_relations);

        printf("\"reconstruct_graph\":{");
        int len = graph.reconstruct_graph.size();
        for (int i = 0; i<len; i++){
            printf("%d:[", graph.reconstruct_graph[i]);
            for (int j = 1; j<=n; j++) write(shuffleDag[flag_generated_graph[graph.reconstruct_graph[i]]].order_list[j]), putchar(',');
            printf("],");
        }putchar('}');

        printf("\"MEC_graph\":");jump_single_vector(graph.MEC_graph_idx);

        printf("\"pair_relations\":[");
        for (int i = 1; i<=n; i++){
            putchar('[');
            for (int j=1;j<=n;j++)printf("%d,",graph.pair_relations[i][j]);
            putchar(']');putchar(',');
        }putchar(']');
        putchar('}');puts("");

        printf("\"pair_relations_count\":[");
        for (int i = 1; i<=n; i++){
            putchar('[');
            for (int j=1;j<=n;j++){
                putchar('[');
                for (int k=0;k<7;k++) printf("%d,",graph.pair_relation_count[i][j][k]);
                putchar(']');putchar(',');
            }
            putchar(']');putchar(',');
        }putchar(']');
        putchar('}');puts("");
    }
    void jump_final_answer(){
        freopen("/Users/lvzhiheng/causal_relation_new_n=6.jsonl", "w", stdout);
        for (int i = 1; i <= unique_DAG_num; i++){
            jump_single_graph(uniqueDag[i]);
        }
    }
}
int main() {
    power_3[0]=1;
    for (int i=1;i<=9;i++)power_3[i]=power_3[i-1]*3;
    node_relations::generate_node_relations();
    json_output::jump_final_answer();
    return 0;
}
