class Constants:
    data_folder = 'data/'
    file_causal_relation = data_folder + 'raw_graphs/causal_relation_n={num_nodes}.jsonl'
    file_out_template = data_folder + 'causalnli_{num_nodes}nodes.json'
    file_split_template = data_folder + '{}.json'


def num_samples2splits(num_samples):
    num_test = min(1000, num_samples // 10)
    num_dev = num_test
    if num_samples < 1000:
        num_test = num_samples // 2
        num_dev = num_samples - num_test
    return {'test': num_test, 'dev': num_dev, 'train': num_samples - num_test - num_dev}


class StatsCalculator:
    def df2stats_with_test(self, df, test_df):

        this_stats = self.df2stats(df)
        test_stats = self.df2stats(test_df)
        test_stats = {'Test_' + k: v for k, v in test_stats.items()}
        this_stats.update(test_stats)
        return this_stats

    def df2stats(self, df):
        import pandas as pd
        df = pd.DataFrame(df)
        df.describe()
        from collections import Counter
        rels = df['relation']
        cnt = Counter(rels)

        stats = {
            f'% {label.capitalize()}': round(cnt[label] / len(rels) * 100, ndigits=2)
            for label in ['entailment', 'contradiction', 'neutral']
        }
        return stats

        #### about premise and hypothesis
        import nltk
        from efficiency.function import avg, flatten_list

        premises = df['premise']
        premises_toks = [nltk.word_tokenize(i) for i in premises]
        premises_len = avg([len(i) for i in premises_toks])

        hyps = df['hypothesis']
        hyps_toks = [nltk.word_tokenize(i) for i in hyps]
        hyps_len = avg([len(i) for i in hyps_toks])

        all_toks = premises_toks + hyps_toks
        all_toks = flatten_list(all_toks)
        vocab_size = len(set(all_toks))

        stats.update({
            '# Tokens/Prem.': premises_len,
            '# Tokens/Hypo.': hyps_len,
            'Vocab Size': vocab_size,
        })
        return stats


def main():
    from efficiency.log import fread
    from efficiency.function import set_seed

    set_seed()
    import pandas as pd
    stats = []
    stats_calculator = StatsCalculator()

    all_data = []
    all_data_test = []
    for num_nodes in range(2, 7):
        file = C.file_out_template.format(num_nodes=num_nodes)
        data = fread(file)
        df = []
        for gold in data:
            relation = gold['relation']
            gold_rel = 'yes' if relation == 'entailment' else "no"
            df.append(gold)

        num_samples = len(df)
        split_sizes = num_samples2splits(num_samples)

        all_data.extend(df)
        all_data_test.extend(df[:split_sizes['test']])

        df = pd.DataFrame(df)

        this_stats = {
            'Num Nodes': num_nodes,
            '# Samples': num_samples,
            '# Test': split_sizes['test'],
            '# Dev': split_sizes['dev'],
            '# Train': split_sizes['train']}
        this_stats.update(stats_calculator.df2stats_with_test(df, df[:split_sizes['test']]))

        stats.append(this_stats)

        df = pd.DataFrame(stats).transpose()
        print(df)
        print(df.style.to_latex())

    df = all_data
    num_samples = len(df)
    df = pd.DataFrame(df)

    this_stats = {
        'Num Nodes': 'All',
        '# Samples': num_samples,
    }
    this_stats.update({f'# {split}': sum(i[f'# {split}'] for i in stats)
                       for split in ['Test', 'Dev', 'Train']})
    this_stats.update(stats_calculator.df2stats_with_test(df, all_data_test))

    stats.append(this_stats)

    df = pd.DataFrame(stats).transpose()
    print(df)
    print(df.style.to_latex())

    #### word-correlation analysis


if __name__ == '__main__':
    C = Constants()
    main()
