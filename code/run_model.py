class Constants:
    data_folder = 'data/'
    file_causal_relation = data_folder + 'raw_graphs/causal_relation_n={num_nodes}.jsonl'

    file_out_template = data_folder + 'data_3class/causalnli_{num_nodes}nodes.json'

    file_split_csv_template = data_folder + 'binary_classification_{}.csv'

    variable_refactor = True
    folder_output = data_folder + 'outputs/'
    if variable_refactor:
        folder_output += 'variable_refactor/'
        data_folder += 'data_3class_from_Z/'
    else:
        data_folder += 'data_3class/'

    file_split_json_template = data_folder + '{}.json'

    file_output_template = folder_output + '{}_test.csv'
    file_all_preds = file_output_template.format('all')
    file_prompt2response_template = folder_output + 'prompt2response_lookup_{}.json'
    finetune_input_file_tmpl = data_folder + 'tmp/causalnli_{ft_loss}_{split}.jsonl'

    model_name2model_name_full = {
        'bert_base_mnli': 'textattack/bert-base-uncased-MNLI',
        'roberta_mnli': "roberta-large-mnli",

        'deberta_xlarge_mnli': 'microsoft/deberta-xlarge-mnli',
        # 'deberta_large_mnli': 'Narsil/deberta-large-mnli-zero-cls',
        # 'deberta_large_mnli': 'microsoft/deberta-large-mnli',

        'distilbert_mnli': 'typeform/distilbert-base-uncased-mnli',
        'distilbart_mnli': 'valhalla/distilbart-mnli-12-1',
        'bart_large_mnli': 'facebook/bart-large-mnli',

    }
    random_model_name2weights = {
        'random_uniform': [1 / 3, 1 / 3, 1 / 3],
        'random_proportional': [0.1857, 0.5582, 0.2561],
        'random_majority': [0, 1, 0],
    }
    gpt_model_name2engine_name = {
        'gpt_a': 'ada',
        'gpt_b': 'babbage',
        'gpt_c': 'curie',
        'gpt_d': 'davinci',
        'gpt_d001': 'text-davinci-001',
        # 'gpt_d002': 'text-davinci-002',
        'gpt3instruct': 'text-davinci-002',
        'gpt_d003': 'text-davinci-003',
        'gpt_d003cot': 'text-davinci-003',

        'gpt3.5': "gpt-3.5-turbo",
        'gpt4': "gpt-4",

        ## 'gpt_d_gen_ft': 'davinci:ft-academicszhijing:causalnli-dev-2022-10-28-22-32-05',
        ## 'gpt_a_cls_ft': 'ada:ft-academicszhijing:causalnli-cls-dev-2022-10-29-00-09-50',
        ## 'gpt_b_cls_ft': 'babbage:ft-academicszhijing:causalnli-cls-dev-2022-10-29-00-55-53',
        ## 'gpt_c_cls_ft': 'curie:ft-academicszhijing:causalnli-cls-dev-2022-10-29-00-39-19',
        ## 'gpt_d_cls_ft': '',
        ## 'gpt_a_cls2_ft': 'ada:ft-academicszhijing:causalnli-cls2-dev-2022-10-29-01-11-33',
        ## 'gpt_b_cls2_ft': 'babbage:ft-academicszhijing:causalnli-cls2-dev-2022-10-29-01-28-06',
        ## 'gpt_c_cls2_ft': 'curie:ft-academicszhijing:causalnli-cls2-dev-2022-10-29-01-58-07',
        ## 'gpt_d_cls2_ft': '',

        # 'gpt_a_cls_10k_ft': 'ada:ft-academicszhijing:causalnli-cls-10k-2022-10-29-12-08-18',
        # 'gpt_b_cls_10k_ft': 'babbage:ft-academicszhijing:causalnli-cls-10k-2022-10-29-13-10-17',
        # 'gpt_c_cls_10k_ft': 'curie:ft-academicszhijing:causalnli-cls-10k-2022-10-29-12-34-57',
        # 'gpt_d_cls_10k_ft': 'davinci:ft-academicszhijing:causalnli-cls-10k-2022-11-01-12-44-59',

        'gpt_a_cls_1k_ft': 'ada:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-54-29',
        'gpt_b_cls_1k_ft': 'babbage:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-40-30',
        'gpt_c_cls_1k_ft': 'curie:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-07-11',
        'gpt_d_cls_1k_ft': 'davinci:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-31-58',
    }
    gpt_model_name2engine_name = {
        'gpt_a_cls_1k_ft': 'ada:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-54-29',
        'gpt_b_cls_1k_ft': 'babbage:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-40-30',
        'gpt_c_cls_1k_ft': 'curie:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-07-11',
        'gpt_d_cls_1k_ft': 'davinci:ft-causalnlp-api:causalnli-cls-1k-2023-05-11-13-31-58',
    }
    models_from_coauthors = [
        # 'bert_base',
        # 'bert_large',
        # 'roberta_base',
        # 'roberta_large',
        # 'longformer_base',
        #
        # 'bert_base_mnli_ft',
        # 'roberta_large_mnli',

        'bert_base_mnli_ft',
        'bert_base_mnli',
        'bert_base',
        'bert_large_mnli',
        'bert_large',
        'deberta_large_mnli',
        'deberta_xlarge_mnli',
        'distilbart_mnli',
        'distilbert_mnli_42.06',
        'distilbert_mnli',
        'huggingface_mnli',
        'longformer_base',
        'random_majority',
        'random_proportional',
        'random_uniform',
        'roberta_base_mnli',
        'roberta_base',
        'roberta_large_mnli',
        'roberta_large',
        'roberta_mnli',
    ]
    models_from_coauthors = [
        'llama030',
        'llama013',
        'llama065',
        'llama007',
        'alpaca007',
    ]
    models_from_coauthors = [
        'roberta_large_mnli',
    ]
    ft_loss2suffix = {
        'gen': 'causalnli-dev',
        'cls': 'causalnli-cls-',
        'cls2': 'causalnli-cls2-',
    }
    pred_rel_lookup = {
        'necessarily true': 'entailment',
        'necessarily false': 'contradiction',
        'neither to say': 'neutral',
        'yes': 'entailment',
        'no': 'contradiction',
        'true': 'entailment',
        'false': 'contradiction',
        'neither': 'neutral',

        'must be true': 'entailment',
        'must be false': 'contradiction',
        'not enough information': 'neutral',

        'neutral': 'contradiction',  # TODO: comment this out if you are doing 3-way NLI.
        'not necessarily true': 'contradiction',
    }
    gold_rel2gpt_completion = {
        'entailment': 'necessarily true".',
        'contradiction': 'necessarily false".',
        'neutral': 'neither".',
    }
    gold_rel2gpt_cls = {
        'entailment': ' true',
        'contradiction': ' false',
        'neutral': ' maybe',
    }
    gold_rel2gpt_cls2 = {
        'entailment': ' true',
        'contradiction': ' not necessarily true',
        'neutral': ' not necessarily true',
    }

    classes = ['entailment', 'contradiction', 'neutral']

    pred_rels = ['must be true', 'must be false', 'neither to say', ]
    options = [i.capitalize() for i in pred_rels]
    options = [f'"{i}"' for i in options]
    options[-1] = 'or ' + options[-1]
    options = ', '.join(options)
    prompt_tmpl_human_like = 'You are a highly intelligent question-answering bot with profound knowledge of causal inference.\n\n' \
                             'Question: {premise}\n' \
                             'Determine the truth value the following statement: {hypothesis}\n' \
                             'The options you may choose from are: ' + options + \
                             '. You only use "Not enough information" when you really don\'t have any idea.' \
                             '\n\nAnswer:'

    prompt_tmpl_direct = 'This is a question to infer causation from correlation by causal inference.\n\n' \
                         'Question: {premise}\nCan we deduct the following: {hypothesis} Just answer "Yes" or "No."\n\nAnswer:'

    prompt_tmpl_direct = 'Question: {premise}' \
                         " Is it necessarily true, necessarily false, or neither to say: " \
                         '{hypothesis}?' + \
                         "\n\nAnswer:"

    prompt_tmpl_direct = 'Question: {premise}\nCan we deduct the following: {hypothesis}? Just answer "Yes" or ' \
                         '"No."\n\nAnswer:'

    prompt_tmpl_human_like = 'Question: {premise}' \
                             " Is it necessarily true, necessarily false, or neither to say: " \
                             "{hypothesis}?" + \
                             "\n\nAnswer: Let's answer step by step."

    prompt_tmpl_human_like_conclusion = \
        'Therefore, the final answer ("necessarily true", "necessarily false", or "neither") is "'
    from efficiency.function import rstrip_word
    prompt_tmpl_finetune_gen = rstrip_word(prompt_tmpl_human_like, "Let's answer step by step.") + \
                               prompt_tmpl_human_like_conclusion.replace('herefore, t', '')
    prompt_tmpl_finetune_cls = '{premise}' \
                               " Is the following statement true, false, or maybe: " \
                               "{hypothesis}?"
    prompt_tmpl_finetune_cls2 = '{premise}' \
                                " Is the following statement true, or not necessarily true: " \
                                "{hypothesis}?"
    prompt_tmpl_generic = 'Premise: {premise}\nHypothesis: {hypothesis}'

    # Question: {premise}
    # What do we know about the following statement:
    # The options you may choose from are: "True", "False", or "Not Enough Information".
    # Answer:

    def __init__(self):
        from glob import glob
        self.model_name2existing_files = lambda model_name: glob(self.folder_output + f'*{model_name}_*')

    def rel_normalize(self, surface_form):
        '''
        Example for ada_ft: "false, maybe? false, maybe?",
                            "maybe false false false false false false false"
        '''
        from nltk import word_tokenize
        rel_normalize = lambda i: self.pred_rel_lookup.get(i, i)

        normalized = rel_normalize(rel_normalize(surface_form))
        if normalized not in self.classes:
            surface_form = ' '.join(word_tokenize(surface_form)[:2])
            normalized = rel_normalize(rel_normalize(surface_form))
            if normalized not in self.classes:
                surface_form = ' '.join(word_tokenize(surface_form)[:1])
                normalized = rel_normalize(rel_normalize(surface_form))
                if normalized not in self.classes:
                    return 'contradiction'
        return normalized


class Model:
    def __init__(self, model_name):
        self.model_output = []
        self.model_name = model_name
        self.prompt2response_file = C.file_prompt2response_template.format(model_name)

        self.clean_pred_func = lambda i: i.lower()

    def set_files(self):
        from efficiency.log import fread

        self.data_input_file = C.file_split_json_template.format('test')
        self.model_output_file = C.file_output_template.format(self.model_name)

        self.data_input = fread(self.data_input_file)
        self.model_output = fread(self.model_output_file)

        from efficiency.log import show_var
        show_var(['len(self.data_input)', 'len(self.model_output)'])

        from efficiency.log import fread
        prompt2response = {}
        for file in C.model_name2existing_files(self.model_name):
            data = fread(file)
            if isinstance(data, dict):
                prompt2response.update(data)
            else:
                if 'pred' in data[0]:
                    prompt2response.update({i['prompt']: i['pred'] for i in data})
                elif 'gpt_prompt' in data[0]:
                    prompt2response.update({i['gpt_prompt']: i['gpt_response'] for i in data})
        self.prompt2response = prompt2response

    def save_prompt2response(self, verbose=False):
        import json
        from efficiency.log import fwrite
        fwrite(json.dumps(self.prompt2response, indent=4), self.prompt2response_file, verbose=verbose)

    def query_text(self, prompt, strip_func=lambda i: i.strip()):
        if isinstance(prompt, str):
            # prompt = str(prompt)
            prompt = prompt.strip()

        if self.inference_is_fast:
            response = self.query_unseen_text(prompt)
            response = strip_func(response)
        else:
            if prompt not in self.prompt2response:
                response = self.query_unseen_text(prompt)
                response = strip_func(response)
                self.prompt2response[prompt] = response
                self.save_prompt2response()

            response = self.prompt2response[prompt]

        return response

    def run_inference(self):
        if self.model_name in set(C.model_name2model_name_full) | set(C.random_model_name2weights):
            self.inference_is_fast = True
        else:
            self.inference_is_fast = False

        from efficiency.log import show_var
        show_var(['self.inference_is_fast'])

        if not hasattr(self, 'query_by_prem_and_hypo'):
            print('[Warning] Skipping inference because the models are not defined')
            return

        self.set_files()
        output_file = self.model_output_file

        import pandas as pd
        from tqdm import tqdm

        data_to_save = []
        for gold in tqdm(self.data_input):
            premise = gold['premise']
            hypothesis = gold['hypothesis']
            response, prompt = self.query_by_prem_and_hypo(premise, hypothesis)

            data_to_save.append({
                "pred": response,
                "gold": gold['relation'],
                "prompt": prompt,
                "id": gold['id'],
            })
            if not self.inference_is_fast:
                df = pd.DataFrame(data_to_save)
                df.to_csv(output_file, index=False)
        if self.inference_is_fast:
            df = pd.DataFrame(data_to_save)
            df.to_csv(output_file, index=False)
        print(f'[Info] Saved {len(df)} entries to {output_file}')

    def run_finetune(self, split=['train', 'dev'][0], ft_loss=['gen', 'cls', 'cls2'][1]):
        # self.set_files()
        output_file = C.finetune_input_file_tmpl.format(ft_loss=ft_loss, split=split)

        if ft_loss == 'cls':
            gold_rel2gpt_label = C.gold_rel2gpt_cls
            prompt_tmpl_finetune = C.prompt_tmpl_finetune_cls
        elif ft_loss == 'cls2':
            gold_rel2gpt_label = C.gold_rel2gpt_cls2
            prompt_tmpl_finetune = C.prompt_tmpl_finetune_cls2
        else:
            gold_rel2gpt_label = C.gold_rel2gpt_completion
            prompt_tmpl_finetune = C.prompt_tmpl_finetune_gen

        split2num_data = {'dev': None, 'train': None}
        num_data = split2num_data[split]

        train_file = C.file_split_json_template.format(split)
        from efficiency.log import fread
        all_data = fread(train_file)

        from efficiency.function import random_sample
        data = random_sample(all_data, num_data)

        from tqdm import tqdm

        data_to_save = []
        import json
        from efficiency.log import fwrite
        for gold in tqdm(data):
            premise = gold['premise']
            hypothesis = gold['hypothesis']
            gold_rel = gold['relation']
            if ft_loss == 'cls2':
                gold_rel = C.rel_normalize(gold_rel)
            gold_rel = gold_rel2gpt_label[gold_rel]

            prompt = prompt_tmpl_finetune.format(premise=premise, hypothesis=hypothesis.strip().rstrip('.'))

            data_to_save.append({
                "prompt": prompt,
                "completion": gold_rel,
                # "id": gold['id'],
            })

        writeout = [json.dumps(i) for i in data_to_save]
        writeout = '\n'.join(writeout)
        fwrite(writeout, output_file, verbose=True)
        # import pandas as pd
        #     df = pd.DataFrame(data_to_save)
        #     df.to_csv(output_file, index=False)
        # print(f'[Info] Saved {len(df)} entries to {output_file}')

    def evaluate(self, detailed=True):
        self.set_files()
        print(self.model_name)

        perf = []

        num_classes = 3
        if C.rel_normalize('neutral') == 'contradiction':
            num_classes = 2

        all_preds = []
        id2pred_n_gold = {}
        for item in self.model_output:
            pred_rel = self.clean_pred_func(item['pred'])
            gold_rel = item['gold']
            pred_rel = C.rel_normalize(pred_rel)
            gold_rel = C.rel_normalize(gold_rel)
            all_preds.append(pred_rel)
            # if pred_rel in C.pred_rel_lookup:
            #     # num_classes = 2
            #     # gold_rel = 'yes' if item['gold'] == 'entailment' else "no"
            #     pred_rel = C.pred_rel_lookup[pred_rel]
            identifiers = item['id'].split('__')
            identifiers = dict([i.split('=', 1) for i in identifiers])

            this_perf = {'pred': pred_rel,
                         'gold': gold_rel,
                         'id': item['id'],
                         }
            id2pred_n_gold[item['id']] = {
                'pred': pred_rel,
                'gold': gold_rel,
            }
            if detailed:
                this_perf.update({
                    'num_nodes': identifiers['num_nodes'],
                    'causal_relation': identifiers['causal_relation'],
                })
            perf.append(this_perf)
        if not perf:
            print('[Warning] The model output file is empty.')
            import pdb;
            pdb.set_trace()
            return {}, {}
        import pandas as pd
        df = pd.DataFrame(perf)


        ## [ sanity check of the GPT response parser
        from collections import Counter
        # preds = [self.clean_pred_func(i['pred']) for i in self.model_output]
        cnt_preds = Counter(all_preds)
        if len(cnt_preds) > num_classes:  # not only "yes" or "no"
            print(cnt_preds)
            writeout = [{'response': k, 'occurrences': v} for k, v in cnt_preds.items()]
            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(writeout, 'data/tmp/response_surface_forms.csv')
            import pdb;
            pdb.set_trace()
        ## ]

        my_report_dicts = []

        my_report_dict = {'subset': 'All'}
        my_report_dict.update(self.perf_df2report_dict(df))
        my_report_dicts.append(my_report_dict)

        if detailed:
            for num_nodes, this_df in df.groupby(['num_nodes']):
                my_report_dict = {'subset': num_nodes}
                my_report_dict.update(self.perf_df2report_dict(this_df))
                my_report_dicts.append(my_report_dict)

            for causal_relation, this_df in df.groupby(['causal_relation']):
                my_report_dict = {'subset': causal_relation}
                my_report_dict.update(self.perf_df2report_dict(this_df))
                my_report_dicts.append(my_report_dict)
        report_df = pd.DataFrame(my_report_dicts)
        pd.set_option('display.max_columns', None)
        if detailed:
            print(report_df)
            import pdb;pdb.set_trace()
            report_df.to_csv('data/tmp/performance.csv')
        print(self.model_name)

        return my_report_dicts[0], id2pred_n_gold

    @staticmethod
    def perf_df2report_dict(df):
        import pandas as pd
        df = pd.DataFrame(df)

        from sklearn.metrics import classification_report

        report = classification_report(df['gold'], df['pred'], digits=4)
        # print(report)
        # import pdb;
        # pdb.set_trace()
        report_dict = classification_report(df['gold'], df['pred'], digits=4, output_dict=True)

        labels = [i for i in ['yes', 'entailment', 'no', 'contradiction', 'neutral'] if i in report_dict]
        report_labels = sorted(set(report_dict.keys()) - {'accuracy', 'macro avg', 'weighted avg', 'micro avg'})
        minority_label = min([(report_dict[i]['support'], i) for i in report_labels])[-1]
        majority_label = max([(report_dict[i]['support'], i) for i in report_labels])[-1]
        label = minority_label
        my_report_dict = {
            'F1': report_dict['weighted avg']['f1-score'],
            'Acc': report_dict['accuracy'],
            'P': report_dict['weighted avg']['precision'],
            'R': report_dict['weighted avg']['recall'],
            'Majo_Acc': report_dict[majority_label]['support']
                        / report_dict['weighted avg']['support'],
            'TotalSamples': report_dict['weighted avg']['support'],
            'Mino_Label': label,
            'Mino_Samples': report_dict[label]['support'],
            'Mino_F1': report_dict[label]['f1-score'],
            'Mino_P': report_dict[label]['precision'],
            'Mino_R': report_dict[label]['recall'],
        }
        my_report_dict = report_dict
        return my_report_dict


class RandomBaseline(Model):
    def __init__(self, model_name='random_uniform'):
        super().__init__(model_name)
        self.classes = C.classes
        self.weights = C.random_model_name2weights[self.model_name]

    def query_by_prem_and_hypo(self, premise, hypothesis):
        prompt = C.prompt_tmpl_generic.format(premise=premise, hypothesis=hypothesis)

        response = self.query_text(prompt)
        return response, prompt

    def query_unseen_text(self, prompt):
        import random
        pred = random.choices(population=self.classes, weights=self.weights, k=1)
        pred = pred[0]
        return pred


class HuggingFace(Model):
    def __init__(self, model_name='huggingface_mnli', if_run_model=False):
        super().__init__(model_name)

        self.model_name_full = C.model_name2model_name_full[self.model_name]

        if not if_run_model: return

        # from transformers import pipeline
        # self.model = pipeline("zero-shot-classification", model=self.model_name, )

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_full)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_full)

    def query_by_prem_and_hypo(self, premise, hypothesis):
        prompt = tuple((premise, hypothesis))

        response = self.query_text(prompt)
        return response, prompt

    def query_unseen_text(self, prompt):
        premise, hypothesis = prompt
        # run through model pre-trained on MNLI
        x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                  truncation='only_first')  # truncate the premise if needed
        # x = x.to(device)
        logits = self.model(x)[0]

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true
        # entail_contradiction_logits = logits[:, [0, 2]]
        contra_neutral_entail_logits = logits[:, [0, 1, 2]]
        probs = contra_neutral_entail_logits.softmax(dim=1)
        label2prob = {
            'contradiction': probs[:, 0],
            'neutral': probs[:, 1],
            'entailment': probs[:, 2],
        }
        order = sorted(label2prob.items(), key=lambda i: i[-1], reverse=True)
        argmax_label = order[0][0]
        return argmax_label

    def query_zeroshot_pipeline(self, premise, hypothesis):
        hypothesis_template = "{}"
        results = self.model([premise], [hypothesis], hypothesis_template=hypothesis_template,
                             multi_label=False
                             )
        predicted_label = {results[0]["labels"][0]}  # [0]

    def query_api(self, api_key_name='HUGGINGFACE_API_KEY', ):
        import os
        self.api_key = os.environ[api_key_name]

        api_key = self.api_key

        from huggingface_hub.inference_api import InferenceApi
        inference = InferenceApi(repo_id="bert-base-uncased", token=api_key)
        inference(inputs="The goal of life is [MASK].")
        import pdb;
        pdb.set_trace()

        inference = InferenceApi(repo_id="deepset/roberta-base-squad2", token=api_key)
        inputs = {"question": "Where is Hugging Face headquarters?",
                  "context": "Hugging Face is based in Brooklyn, New York. There is also an office in Paris, France."}
        inference(inputs)

        inference = InferenceApi(repo_id="typeform/distilbert-base-uncased-mnli", token=api_key)
        inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
        params = {"candidate_labels": ["refund", "legal", "faq"]}
        inference(inputs, params)

        inference = InferenceApi(repo_id="paraphrase-xlm-r-multilingual-v1",
                                 task="feature-extraction",
                                 token=api_key,
                                 )


class GPT(Model):
    def __init__(self, model_name='gpt3instruct', api_key_name='OPENAI_API_KEY', default_max_tokens=32):
        super().__init__(model_name)

        import os
        self.api_key = os.environ[api_key_name]

        import openai
        openai.api_key = self.api_key
        self.model = openai
        self.engine_name = C.gpt_model_name2engine_name[model_name]

        from efficiency.nlp import Chatbot
        self.chat = Chatbot(
            system_prompt='You are a highly intelligent question-answering bot with profound knowledge of causal inference.',
            output_file=f'data/tmp/cache_{model_name}_responses.csv')

        from efficiency.function import lstrip_word
        self.clean_pred_func = lambda i: lstrip_word(i.lower().strip().strip(' ".:').strip(), 'that there is ')

        self.engine_finetuned = self.engine_name not in \
                                {'text-davinci-001', 'text-davinci-002', 'text-davinci-003', 'ada', 'gpt-3.5-turbo',
                                 'gpt-4', 'ada', 'babbage', 'curie', 'davinci',
                                 }

        self.engine_ft_loss2suffix = {
            ft_loss: (f':{suffix}2022' in self.engine_name) or
                     (f':{suffix}10k-2022' in self.engine_name) or
                     (f':{suffix}1k-2023' in self.engine_name)
            for ft_loss, suffix in C.ft_loss2suffix.items()
        }

        self.max_tokens = default_max_tokens
        self.stop_symbols = None

        if self.engine_finetuned:
            self.max_tokens = 8
            self.stop_symbols = ['\n\n']
            if self.engine_ft_loss2suffix['gen']:
                self.stop_symbols = ['".', '\n\n', ]

    def query_by_prem_and_hypo(self, premise, hypothesis):
        if self.engine_finetuned:
            if self.engine_ft_loss2suffix['gen']:
                prompt_tmpl = C.prompt_tmpl_finetune_gen
            elif self.engine_ft_loss2suffix['cls2']:
                prompt_tmpl = C.prompt_tmpl_finetune_cls2
            elif self.engine_ft_loss2suffix['cls']:
                prompt_tmpl = C.prompt_tmpl_finetune_cls
            else:
                import pdb;
                pdb.set_trace()
            prompt = prompt_tmpl.format(premise=premise, hypothesis=hypothesis.strip().rstrip('.'))
            response = self.query_text(prompt)
            return response, prompt

        elif 'cot' not in self.model_name:
            prompt = C.prompt_tmpl_direct.format(premise=premise, hypothesis=hypothesis.strip().rstrip('.'))
            response = self.query_text(prompt)
            return response, prompt
        else:
            prompt = C.prompt_tmpl_human_like.format(premise=premise, hypothesis=hypothesis.strip().rstrip('.'))
            prompt_conclusion = C.prompt_tmpl_human_like_conclusion

            response = self.query_text(prompt, strip_func=str)
            new_prompt = prompt + response + '\n\n' + prompt_conclusion

            # example_output = ' "neither".'
            response = self.query_text(new_prompt)
            return response, new_prompt

    def query_unseen_text(self, prompt, max_tokens=None):
        '''
        Costs:
        - 20221010 40USD for all test sets.
        - 20221019 15.58USD for all test sets.
        '''

        if max_tokens is None:
            max_tokens = self.max_tokens
            if prompt.endswith('answer step by step.'):
                max_tokens = 256
            elif prompt.endswith(C.prompt_tmpl_human_like_conclusion[10:]) or prompt.endswith('Answer:'):
                max_tokens = 2
                max_tokens = 10

        response = self.chat.ask(prompt, engine=self.engine_name, max_tokens=max_tokens,
                                 stop_sign=self.stop_symbols, enable_pdb=False)

        return response

    def finetune_commands(self):
        # Reference: https://harishgarg.com/writing/how-to-fine-tune-gpt-3-api/
        '''
        [2022-10-28 20:45:05] Created fine-tune: ft-1FBYAFM3noxALTan0P6UcAoo
        [2022-10-28 20:45:21] Fine-tune costs $101.98
        [2022-10-28 20:45:21] Fine-tune enqueued
        '''
        import os

        data_input_file = C.finetune_input_file

        response = self.model.File.create(
            file=open(data_input_file),
            purpose='fine-tune'
        )
        import pdb;
        pdb.set_trace()
        print(response)

        # Check commands in code/model_gpt3_finetune*.txt


def main(if_run_model=False):
    from efficiency.log import show_var
    show_var(['C.variable_refactor', 'if_run_model'])
    import pdb;
    pdb.set_trace()
    from efficiency.function import set_seed
    set_seed()

    report_dicts = []

    all_models = []

    # for model_name in C.random_model_name2weights:
    #     model = RandomBaseline(model_name=model_name)
    #     all_models.append(model)

    # for model_name, engine_name in list(C.gpt_model_name2engine_name.items()):  # [-4:]:
    #     if not engine_name:
    #         continue
    #     model = GPT(model_name=model_name)
    #     print(model_name, engine_name)
    #     all_models.append(model)

    # for model_name in C.model_name2model_name_full:
    #     model = HuggingFace(model_name=model_name, if_run_model=if_run_model)
    #     all_models.append(model)
    #
    for model_name in C.models_from_coauthors:
        model = Model(model_name)
        all_models.append(model)

    from collections import defaultdict
    id2all_pred_n_gold = defaultdict(dict)
    for model in all_models:
        # model.run_finetune()
        # return
        # model.finetune_commands()

        if if_run_model:
            model.run_inference()
            continue
        report_dict, id2this_pred_n_gold = model.evaluate()
        report_dicts.append((model.model_name, report_dict))

        for id, this_pred_n_gold in id2this_pred_n_gold.items():
            this_pred = this_pred_n_gold['pred']
            id2all_pred_n_gold[id][model.model_name] = this_pred

            # Sanity check:
            this_gold = this_pred_n_gold['gold']
            if 'gold' not in id2all_pred_n_gold[id]:
                id2all_pred_n_gold[id]['gold'] = this_gold
                # id2all_pred_n_gold[id]['id'] = id
            else:
                if id2all_pred_n_gold[id]['gold'] != this_gold:
                    from efficiency.log import show_var
                    show_var(['this_gold', "this_pred_n_gold['id']"])
                    import pdb;
                    pdb.set_trace()

    for id, v in id2all_pred_n_gold.items():
        v['id'] = id
    from efficiency.log import write_dict_to_csv
    write_dict_to_csv(list(id2all_pred_n_gold.values()), C.file_all_preds, verbose=True)

    raw_stats = [(model_name if not C.variable_refactor else model_name + '-VR',
                  i['entailment']['f1-score'], i['entailment']['precision'], i['entailment']['recall'],
                  # i['contradiction']['f1-score'], i['contradiction']['precision'], i['contradiction']['recall'],
                  # i['weighted avg']['f1-score'], i['macro avg']['f1-score'],
                  i['accuracy']
                  )
                 for model_name, i in report_dicts]
    stats = [' & '.join(list(map(lambda a: f"{round(a * 100, 2):.2f}"
    if isinstance(a, float) else a, i))) + ' \\\\ \n'
             for i in raw_stats]
    stats = ''.join(stats)
    stats = stats.replace('_', '-')
    print(stats)

    import pdb;
    pdb.set_trace()

    return

    from efficiency.log import fread
    from efficiency.function import avg
    import pandas as pd
    for num_nodes in range(2, 7):
        file = C.file_out_template.format(num_nodes=num_nodes)
        data = fread(file)
        data_to_save = []
        file_to_save = C.file_to_save.format(num_nodes)
        data_to_save = fread(file_to_save)
        accs = []

        for gold, pred in zip(data, data_to_save):
            relation = gold['relation']
            pred_rel = pred['gpt_response']

            gold_rel = 'yes' if relation == 'entailment' else "no"
            pred_rel = pred_rel.lower()

            accs.append(gold_rel == pred_rel)

        print(avg(accs, decimal=4), len(accs))
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    C = Constants()
    main()
