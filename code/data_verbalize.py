# Set random seed
import random
import numpy as np
random.seed(0)
np.random.seed(0)

class Constants:
    root = '../data'
    file_causal_relation = root + '/causal_relation_new_n={num_nodes}.jsonl'
    file_out_template_var1_var2 = root + '/causalnli_{num_nodes}nodes_var1_var2.json'

    variable_refactor = True
    data_folder_suffix = '_from_Z' if variable_refactor else ''

    data_folder = f'{root}/data_3class{data_folder_suffix}/'
    file_split_json_template = data_folder + '{}.json'
    file_out_template = data_folder + 'causalnli_{num_nodes}nodes.json'

    file_split_csv_template = f'{root}/data_2class{data_folder_suffix}/' + 'data_binary_format/{}.csv'

    @staticmethod
    def powerset(iterable):
        from itertools import chain, combinations

        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


class DataCompiler:
    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes

    def fread_jsonl(self, path, verbose=True):
        if verbose:
            print(f'[Info] Reading {path}')
        json_list = []
        with open(path, 'r', encoding='UTF-8') as f3:
            lines = f3.readlines()
            last_line = None
            for line in lines:
                if last_line == None:
                    last_line = line
                else:
                    new_line = last_line[:-2] + ',' + line
                    last_line = None
                    new_line = ']'.join(new_line.split(',]'))
                    new_line = '}'.join(new_line.split(',}'))
                    new_line = '},"'.join(new_line.split('}"'))
                    json_list.append(eval(new_line))
        return json_list

    def preprocess_json(self, jsonl):
        # Adapted from Zhiheng Lyu's codes to parse the data: https://colab.research.google.com/drive/1hnJfcZACPZgkZZ8kXnbQy8QAH92FtUk2?usp=sharing#scrollTo=WJkomw3LKsdR
        n = self.num_nodes

        def generate_matrix():
            matrix = []
            for i in range(n):
                matrix.append([0] * n)
            return matrix

        def generate_adgacency_matrix(index_3):
            # print(index_3)
            matrix = generate_matrix()
            base = 1
            for i in range(0, n - 1):
                for j in range(i + 1, n):
                    current = index_3 // base % 3
                    # print(current)
                    if current == 1:
                        matrix[i][j] = 1
                    elif current == 2:
                        matrix[j][i] = 1
                    base *= 3
            return matrix

        def shuffle_adgancency_matrix(old_matrix, shuffle_list):
            matrix = generate_matrix()
            for i in range(0, n):
                for j in range(0, n):
                    if (old_matrix[i][j]):
                        matrix[shuffle_list[i - 1]][shuffle_list[j - 1]] = 1
            return matrix

        def shuffle_linklist(linklist, shuffle_list):
            new_linklist = []
            for i in linklist:
                new_linklist.append([shuffle_list[i[0] - 1], shuffle_list[i[1] - 1]])
            return new_linklist

        def generate_all_graph(json_list):
            graph_list = []
            father_list = []
            length = 0
            for i in json_list:
                current_graph = length
                for graph in i['reconstruct_graph']:
                    # graph_list.append()
                    new_graph = {}
                    new_graph['adgancency_matrix'] = generate_adgacency_matrix(graph)
                    new_graph['edge_list'] = shuffle_linklist(i['graph_edges'], i['reconstruct_graph'][graph])
                    graph_list.append(new_graph)
                    father_list.append(current_graph)
                    length += 1
            return graph_list, father_list

        import numpy as np

        def decode_graph(x):
            li = []
            for i in range(n):
                if (x % 2 == 1):
                    li.append(i + 1)
                # li.append(x%2)
                x //= 2
            return li

        def generate_MEC(json_list, graph_list, father_list):
            flag = [0] * len(graph_list)
            MEC_list = []
            for i in json_list:
                MEC_idx = i['MEC_graph']
                flag_qwq = True
                for idx in MEC_idx:
                    if flag[father_list[idx - 1]] == True:
                        flag_qwq = False
                if (not flag_qwq):
                    continue
                MEC = {}
                MEC['indics_of_causal_graph_in_this_MEC'] = MEC_idx
                MEC['relations_between_two_nodes'] = np.array(i['pair_relations_count']) / len(MEC_idx)
                MEC['CI_relation'] = []
                for j in i['CI_relations']:
                    MEC['CI_relation'].append([decode_graph(j[0]), decode_graph(j[1])])
                MEC_list.append(MEC)
                for idx in MEC_idx:
                    flag[father_list[idx - 1]] = True
            return MEC_list

        graph_list, father_list = generate_all_graph(jsonl)
        MEC_list = generate_MEC(jsonl, graph_list, father_list)
        MEC_list = self.simplify_CI_for_MEC_list(MEC_list, n)

        all_data = {
            'num_nodes': self.num_nodes,
            'raw_data': jsonl,
            'all_causal_graphs': graph_list,
            'all_MECs': MEC_list
        }
        return all_data

    def simplify_CI_for_MEC_list(self, MEC_list, n):
        def encode_2bit(li):
            x = 0
            for i in li:
                x += 2 ** (i - 1)
            return x

        def decode_2bit(x):
            li = []
            for i in range(n):
                if (x % 2 == 1):
                    li.append(i + 1)
                x //= 2
            return li

        def find_supperset(x, y, z, power_n, li):
            num_0, num_1 = 0, 0
            for i in range(power_n):
                if ((i >> x) & 1 == 0) and ((i >> y) & 1 == 0) and (i & z == z):
                    if (li[i] == 0):
                        num_0 += 1
                    else:
                        num_1 += 1
            # if (x==1 and y == 2):
            #  print("sup", z, num_0, num_1)
            if num_0 == 0:
                return 1
            elif num_1 == 0:
                return 0
            else:
                return 2

        def check_subset(x, y, z, power_n, li):
            num_0, num_1 = 0, 0
            for i in range(power_n):
                if ((i >> x) & 1 == 0) and ((i >> y) & 1 == 0) and (i & z == i):
                    if i == z:
                        continue
                    if (li[i] == 0):
                        num_0 += 1
                    elif (li[i] == 1):
                        num_1 += 1
            return num_0, num_1

        import numpy as np

        # 格式: CI_relation[x,y,[ind_set],"Atomic"/"Non-Atomic"]; CR_relation[]x,y,[ind_set],"Atomic"/"Non-Atomic"
        def post_process_new_relation(new_CI_relation):
            CI_relation, CR_relation = [], []
            Atomic_info = ["Atomic", "Atomic", "Non-Atomic"]
            for i in new_CI_relation:
                if i[3] == 1:
                    CI_relation.append([i[0], i[1], i[2], Atomic_info[i[4]]])
                else:
                    CR_relation.append([i[0], i[1], i[2], Atomic_info[i[4]]])
            return CI_relation, CR_relation

        def process_CI_relation(n, CI_relation):
            power_n = 2 ** n
            CI_list = np.zeros((n, n, power_n)).tolist()
            CI_supperlist_info = np.zeros((n, n, power_n)).tolist()
            for i in CI_relation:
                # print(i)
                x, y, z = i[0][0], i[0][1], i[1]
                z_num = encode_2bit(z)
                # print(x,y,z_num)
                CI_list[x - 1][y - 1][z_num] = 1
            for i in range(n):
                for j in range(n):
                    for k in range(power_n):
                        CI_supperlist_info[i][j][k] = find_supperset(i, j, k, power_n, CI_list[i][j])
            new_CI_relation = []
            for i in range(n):
                for j in range(n):
                    for k in range(power_n):
                        if ((k & (1 << (i)) != 0) or (k & (1 << (j)) != 0) or i >= j):
                            continue
                        num_0, num_1 = check_subset(i, j, k, power_n, CI_supperlist_info[i][j])
                        # if (i==1 and j==2):
                        #  print(k, num_0, num_1, CI_supperlist_info[i][j])
                        if num_0 > 0 and CI_list[i][j][k] == 0 or num_1 > 0 and CI_list[i][j][k] == 1:
                            continue
                        new_CI_relation.append(
                            [i + 1, j + 1, decode_2bit(k), CI_list[i][j][k], CI_supperlist_info[i][j][k]])
            return post_process_new_relation(new_CI_relation)  # , CI_supperlist_info, CI_list

        # Sample code about how to use it
        for i in range(len(MEC_list)):
            CI_relation, CR_relation = process_CI_relation(n, MEC_list[i]['CI_relation'])
            CI_relation = [i[:3] for i in CI_relation]
            CR_relation = [i[:3] for i in CR_relation]
            MEC_list[i]['simplication_CI_list'], MEC_list[i]['simplication_CR_list'] = CI_relation, CR_relation

        return MEC_list

    def read_causal_graphs(self):

        data = self.fread_jsonl(C.file_causal_relation.format(num_nodes=self.num_nodes))
        all_data = self.preprocess_json(data)
        return all_data
        # data = {"graph_index_2": 0, "graph_index_3": 0, "idx": 1, "graph_edges": [],
        #         "CI_relations": [[3, 0], [3, 4], [5, 0], [5, 2], [6, 0], [6, 1], ],
        #         "reconstruct_graph": {0: [1, 2, 3, ], }, "MEC_graph": [1, ],
        #         "pair_relations": [[0, 0, 0, ], [0, 0, 0, ], [0, 0, 0, ], ]}

    @staticmethod
    def compile_train_dev_test():

        import json
        from collections import defaultdict
        from efficiency.log import fread, fwrite, write_dict_to_csv
        from data_stats import num_samples2splits

        split2data_json = defaultdict(list)

        for num_nodes in range(2, 7):
            file = C.file_out_template.format(num_nodes=num_nodes)
            data = fread(file)

            num_samples = len(data)
            split_sizes = num_samples2splits(num_samples)
            split_start_ix = 0
            for split, split_size in split_sizes.items():
                split_end_ix = split_start_ix + split_size
                split2data_json[split].extend(data[split_start_ix:split_end_ix])
                split_start_ix = split_end_ix

        print({k: len(v) for k, v in split2data_json.items()})

        for split, data_json in split2data_json.items():
            data_csv = []
            for datum_json in data_json:
                relation = datum_json['relation']
                binary_relation = int(relation == 'entailment')

                datum_csv = {
                    'input': f"Premise: {datum_json['premise']}\nHypothesis: {datum_json['hypothesis']}",
                    'label': binary_relation,
                }

                data_csv.append(datum_csv)

            fwrite(json.dumps(data_json, indent=4), C.file_split_json_template.format(split), verbose=True)
            write_dict_to_csv(data_csv, C.file_split_csv_template.format(split), verbose=True)


class Verbalizer:
    properties = ["parent", "non-parent ancestor", "child", "non-child descendant", "has_collider",
                  "has_confounder", "mixed_type"]
    property2hyp_template_original = {
        "parent": "{node_i} directly causes {node_j}.",
        "non-parent ancestor": "{node_i} causes something else which causes {node_j}.",
        "child": "{node_j} directly causes {node_i}.",
        "non-child descendant": "{node_j} is a cause for {node_i}, but not a direct one.",
        "has_collider": "There exists at least one collider (i.e., common effect) of {node_i} and {node_j}.",
        "has_confounder": "There exists at least one confounder (i.e., common cause) of {node_i} and {node_j}.",
    }
    property2hyp_template = {
        "parent": "{node_i} directly affects {node_j}.",
        "non-parent ancestor": "{node_i} influences {node_j} through some mediator(s).",
        "child": "{node_j} directly affects {node_i}.",
        "non-child descendant": "{node_j} influences {node_i} through some mediator(s).",
        "has_collider": "{node_i} and {node_j} together cause some other variable(s).",
        "has_confounder": "Some variable(s) cause(s) both {node_i} and {node_j}.",
    }

    def __init__(self, all_data, template = "original"): # template = "original" or "paraphrased"
        self.all_data = all_data

        self.list2text = lambda my_list: " and ".join(
            [", ".join(my_list[:-1]), my_list[-1]] if len(my_list) > 2 else my_list)
        if not C.variable_refactor:
            self.node_ix2surface_form = lambda i: chr(i + 64)  # 1 --> a
        else:
            self.node_ix2surface_form = lambda i: chr(91 - i)  # 1 --> Z

        self.prob2label = lambda prob: 'entailment' if prob == 1 else \
            'contradiction' if prob == 0 else \
                'neutral'
        self.template = template
        if template == "original":
            self.property2hyp_template_used = self.property2hyp_template_original
        else:
            self.property2hyp_template_used = self.property2hyp_template

    def raw_data2nli_format(self):
        all_data = self.all_data
        num_nodes = all_data['num_nodes']
        raw_data = all_data['raw_data']
        all_causal_graphs = all_data['all_causal_graphs']
        all_MECs = all_data['all_MECs']

        from itertools import combinations, permutations
        all_one_node_set = set(range(1, num_nodes + 1))
        all_two_nodes_set = set(combinations(all_one_node_set, 2))
        all_two_nodes_list = set(permutations(all_one_node_set, 2))
        powerset = C.powerset(all_one_node_set)

        ordered_mecs = list(enumerate(all_MECs))

        from efficiency.function import set_seed
        set_seed(1)  # if the seed=0, then 3nodes wouldn't have entailment in the test set.
        import random
        random.shuffle(ordered_mecs)

        nli_data = []
        for mec_ix, mec_info in ordered_mecs:
            cis = mec_info['CI_relation']
            # print(f"mec_ix={mec_ix}, cis={cis}")
            # TODO: we could perhaps also verbalize 'simplication_CI_list', 'simplication_CR_list'

            two_nodes_n_relation_ix2prob = mec_info['relations_between_two_nodes']
            from collections import defaultdict
            two_nodes2conds = defaultdict(list)
            for two_nodes, condition in cis:
                two_nodes2conds[tuple(two_nodes)].append(condition)
            two_nodes2conds = {k: sorted(v, key=lambda i: len(i)) for k, v in two_nodes2conds.items()}

            # For two-variable relations, describe by corrs; for >=3-variable relations, describe by CITs
            corrs = []
            for two_nodes in sorted(all_two_nodes_set):
                uncond_ind = False
                if two_nodes in two_nodes2conds:
                    if not two_nodes2conds[two_nodes][0]:  # i.e., cond_set = [], i.e., unconditional independence
                        uncond_ind = True
                if not uncond_ind:
                    corrs.append(two_nodes)

            cond_inds = [list(k) + [cond]  # TODO: check CI redundancy with Zhiheng; check correlation mentions too
                         for k, conds in two_nodes2conds.items() for cond in conds]
            cond_inds = sorted(cond_inds)
            # hyp_2nodes = sorted(all_two_nodes_set)
            hyp_2nodes = sorted(all_two_nodes_list)
            nli_data.extend(self.symbol2nli(corrs, cond_inds, hyp_2nodes,
                                            two_nodes_n_relation_ix2prob, mec_ix, num_nodes))
        import json
        from efficiency.log import fwrite
        # print(C.file_out_template.format(num_nodes=num_nodes))
        # raise NotImplementedError
        fwrite(json.dumps(nli_data, indent=4), C.file_out_template.format(num_nodes=num_nodes), verbose=True)

        return
        # for causal_graph in all_causal_graphs:
        #     adj = np.array(causal_graph['adgancency_matrix'])
        #     node2parents = {}
        #     for node_i in range(num_nodes):
        #         parents_binary = np.array(adj)[:, node_i]
        #         parents_ixs = np.concatenate(np.where(parents_binary == 1))
        #         if len(parents_ixs):
        #             node2parents[node_i + 1] = (parents_ixs + 1).tolist()  # our data is 1-based
        #
        #     nli_data.extend(self.symbol2nli_from_causal_graph(node2parents, all_one_node_set))

        import json
        from efficiency.log import fwrite
        fwrite(json.dumps(nli_data, indent=4), C.file_out_template_var1_var2.format(num_nodes=num_nodes), verbose=True)

    def symbol2nli_from_causal_graph(self, node2parents, all_one_node_set, num_contradicts=1):
        node2parents = {self.node_ix2surface_form(k): [self.node_ix2surface_form(i) for i in v]
                        for k, v in node2parents.items()}
        all_one_node_set = [self.node_ix2surface_form(i) for i in all_one_node_set]

        var_set = self.list2text(sorted(all_one_node_set))
        premise = f"Suppose we know the causal graph among the variables {var_set}, where "
        if not node2parents:
            premise += "all the variables are independent of each other."
        else:
            premise += "all the causal relations are the following: "
            premise += ' '.join([f"{node_i} directly causes {node_j}."
                                 for node_i, parents in node2parents.items()
                                 for node_j in parents])

        nli_data = []
        for node_i in all_one_node_set:
            for node_j in set(all_one_node_set) - {node_i}:
                label = self.prob2label(node_j in node2parents.get(node_i, {}))
                hyp = f"No matter which variables we control for, changing {node_j} " \
                      f"is always a way to change {node_i}."
                nli_data.append({
                    'premise': premise,
                    'hypothesis': hyp,
                    'relation': label,
                })

        from efficiency.function import random_sample, set_seed
        set_seed()
        for node_i, parents in node2parents.items():
            non_self = set(all_one_node_set) - {node_i}
            powerset = set(C.powerset(non_self)) - {tuple(parents)}
            powerset_samples = random_sample(powerset, num_contradicts)

            parents2label = {i: 'contradiction' for i in powerset_samples}
            parents2label[tuple(parents)] = 'entailment'
            for parents, label in parents2label.items():
                non_parents = sorted(non_self - set(parents))
                num_non_parents = len(non_parents)

                if num_non_parents == 0:
                    continue
                elif num_non_parents == 1:
                    pass
                elif num_non_parents >= 3:
                    parents = self.list2text(parents)
                    non_parents = self.list2text(non_parents)

                    hyp_paraphrases = [
                        f"Given {parents}, none of the nodes {non_parents} matters for {node_i}.",
                        f"If {parents} are fixed, {node_i} will not be affected by any of the nodes {non_parents}.",
                        f"The variables {parents} altogether can screen off the correlation between {node_i} and any of the " \
                        f"variables {non_parents}."
                    ]

                    nli_data += [{
                        'premise': premise,
                        'hypothesis': hyp,
                        'relation': label,
                    } for hyp in hyp_paraphrases
                    ]

            # TODO: ask Luigi to proofread all expressions.
        return nli_data

    def symbol2nli(self, corrs, cond_inds, hyp_2nodes, two_nodes_n_relation_ix2prob, mec_ix, num_nodes):
        node_ix2surface_form = self.node_ix2surface_form
        corrs = [[node_ix2surface_form(node_i), node_ix2surface_form(node_j), ] for node_i, node_j in corrs]
        cond_inds = [[node_ix2surface_form(node_i), node_ix2surface_form(node_j),
                      [node_ix2surface_form(i) for i in cond_set]]
                     for node_i, node_j, cond_set in cond_inds]
        hyp_2nodes_str = [[node_ix2surface_form(node_i), node_ix2surface_form(node_j), ] for node_i, node_j in
                          hyp_2nodes]

        # corrs = [["A", "B"], ["A", "C"], ["B", "C"], ]
        # cond_inds = [["A", "C", ["B"]], ]

        corr_statements = [f"{node_i} correlates with {node_j}." for node_i, node_j in corrs]
        cond_ind_statements = [f"{node_i} and {node_j} are independent given {self.list2text(cond_set)}."
                               if cond_set else f"{node_i} is independent of {node_j}."
                               for node_i, node_j, cond_set in cond_inds]
        all_variables = [node_ix2surface_form(node_i) for node_i in range(1, 1 + num_nodes)]

        premise = f"Suppose there is a closed system of {num_nodes} variables, {self.list2text(all_variables)}. All " \
                  f"the statistical relations among these {num_nodes} variables are as follows: "

        if corr_statements:
            premise += f"{' '.join(corr_statements)}"
            if cond_ind_statements:
                premise += f" However, "
        if cond_ind_statements:
            premise += f"{' '.join(cond_ind_statements)}"

        # hyp_2nodes = [["A", "B"], ["A", "C"], ["B", "C"], ]
        nli_data = []
        for (node_i, node_j), (node_i_str, node_j_str) in zip(hyp_2nodes, hyp_2nodes_str):
            for property, hyp_template in self.property2hyp_template_used.items():
                if node_j < node_i:
                    # print(f"mec_ix={mec_ix}, node_i={node_i}, node_j={node_j}, property={property}, node_j is larger than node_i")
                    continue
                # print(f"mec_ix={mec_ix}, node_i={node_i}, node_j={node_j}, property={property}")
                if property == "has_collider" or property == "has_confounder":
                    # random swap node_i and node_j
                    if random.random() < 0.5:
                        hyp = hyp_template.format(node_i=node_j_str, node_j=node_i_str)
                    else:
                        hyp = hyp_template.format(node_i=node_i_str, node_j=node_j_str)
                elif (property == "non-parent ancestor" or property == "non-child descendant") and self.template == "original":
                    if random.random() < 0.5:
                        if property == "non-parent ancestor":
                            hyp_template_new = self.property2hyp_template_used["non-child descendant"]
                        else:
                            hyp_template_new = self.property2hyp_template_used["non-parent ancestor"]
                        hyp = hyp_template_new.format(node_i=node_j_str, node_j=node_i_str)
                    else:
                        hyp = hyp_template.format(node_i=node_i_str, node_j=node_j_str)
                else:
                    hyp = hyp_template.format(node_i=node_i_str, node_j=node_j_str)

                rel_ix = self.properties.index(property)
                # print(two_nodes_n_relation_ix2prob.shape)
                prob = two_nodes_n_relation_ix2prob[node_i - 1, node_j - 1, rel_ix]

                label = self.prob2label(prob)
                nli_data.append({
                    'premise': premise,
                    'hypothesis': hyp,
                    'relation': label,
                    'id': f'num_nodes={num_nodes}__mec_id={mec_ix}__node_i={node_i}__node_j={node_j}__'
                          f'causal_relation={property.replace(" ", "_")}__prob={prob:.2f}',
                })

        return nli_data


def main():
    for num_nodes in range(2, 7):
        data_compiler = DataCompiler(num_nodes=num_nodes)
        all_data = data_compiler.read_causal_graphs()
        # print(all_data.keys())
        # print(len(all_data['raw_data']))
        # print(len(all_data['all_causal_graphs']))
        # print(all_data['all_causal_graphs'][-1])
        # print(len(all_data['all_MECs']))
        # print(all_data['all_MECs'][0])
        # print(all_data['all_MECs'][-1])
        verbalizer = Verbalizer(all_data)
        verbalizer.raw_data2nli_format()
    DataCompiler.compile_train_dev_test()


if __name__ == '__main__':
    C = Constants()
    main()
