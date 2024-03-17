import csv
from efficiency.log import fread, write_dict_to_csv

class PromptComposer:
    label2desc = {
        1: 'The premise provides the necessary conditions for the hypothesis. So if the premise is true, '
           'the hypothesis must be true.',
        0: 'The premise is not a necessary condition for the hypothesis.'
    }
    prompt_template = """Here is a causal inference rule:

###
{premise_and_hypothesis}

Relation between the promise and hypothesis: {relation}
###

Please provide a real-world example instantiating this phenomenon. Format it also as "Premise:", "Hypothesis:", and "Relation between the promise and hypothesis:".
""".strip()

    tag_lookup = {'prem': "Premise:", 'hyp': "Hypothesis:", 'label': "Relation between the premise and hypothesis:", }

    def __init__(self):

        from efficiency.nlp import Chatbot
        self.model = Chatbot(model_version='gpt3.5', max_tokens=256)

    def data2prompt(self, text, label):
        hyp_tag = self.tag_lookup['hyp']
        text = text.replace(f' {hyp_tag}', f'\n\n{hyp_tag}')
        desc = self.label2desc[label]
        prompt = self.prompt_template.format(premise_and_hypothesis=text, relation=desc)
        return prompt

    def get_story(self, prompt):
        story = self.model.ask(prompt)
        print(story)
        text_input = story.split(self.tag_lookup['label'], 1)
        if len(text_input) == 2:
            expl = text_input[1]
            text_input = text_input[0]
        else:
            expl = ""
            text_input = text_input[0]

        hyp_tag = self.tag_lookup['hyp']
        text_input = text_input.replace(f'\n\n{hyp_tag}', f' {hyp_tag}')

        label = None
        for rel, desc in self.label2desc.items():
            if expl.startswith(desc):
                label = rel
                break
        return {
            'input_story': text_input,
            'label_story': label,
            'explanations_story': expl,
        }

class DataManager:
    def generate_story_file(self):
        file_format = 'data/data_2class/{}.csv'
        story_file_format = 'data/data_2class/{}_story.csv'
        self.story_file_format = story_file_format
        for split in ['test', 'dev']:
            file = file_format.format(split)
            story_file = story_file_format.format(split)
            self.process_data(file, story_file)

    def process_data(self, file, story_file, max_n_nodes=4):
        from efficiency.log import fread
        from efficiency.function import lstrip_word
        data = fread(file)

        composer = PromptComposer()
        from tqdm import tqdm
        f_out = open(story_file, "a+")
        writer = csv.writer(f_out, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["input_story", "label_story", "explanations_story"])
        for idx, row in tqdm(enumerate(data)):
            text = row['input']
            label = row['label']
            n_nodes = int(lstrip_word(text, 'Premise: Suppose there is a closed system of ').split(' ', 1)[0])
            if n_nodes > max_n_nodes:
                continue
            prompt = composer.data2prompt(text, label)
            story = composer.get_story(prompt)
            row.update(story)
            writer.writerow([row["input_story"], row["label_story"], row["explanations_story"]])


class PostProcessor:
    def __init__(self):
        self.content = None
        
    def step1_reformat_answer(self, file):
        data = fread(file)
        data2 = fread(file.replace("_story.csv", ".csv"))
        new_data = []
        for row, row2 in zip(data, data2):
            new_row = {}
            new_row['input_story'] = row['input_story'].strip()
            if type(row['explanations_story']) == float:
                new_row['explanations_story'] = ""
            else:
                new_row['explanations_story'] = row['explanations_story'].strip()
            new_row["label_story"] = str(row2['label'])
            new_data.append(new_row)
        self.content = new_data
        
    def step2_make_column(self):
        data = self.content
        new_data = []
        for idx, row in enumerate(data):
            new_row = {}
            try:
                if "Real-world example:" in row['explanations_story']:
                    if "The statistical relations among these variables are as follows" in row['explanations_story']:
                        new_row['premise'] = row['explanations_story'].split("The statistical relations among these variables are as follows",1)[0].split("Premise:",1)[1].strip()
                        new_row['hypothesis'] = row['explanations_story'].split("The statistical relations among these variables are as follows",1)[1].split("Hypothesis:",1)[1].strip()
                        new_row['explanation'] = row['explanations_story'].split("The statistical relations among these variables are as follows",1)[1].split("Hypothesis:")[0].strip()
                    else:
                        new_row['premise'] = row['explanations_story'].split("Real-world example:",1)[1].split("Premise:",1)[1].split("Hypothesis:")[0].strip()
                        new_row['hypothesis'] = row['explanations_story'].split("Real-world example:",1)[1].split("Hypothesis:",1)[1].split("Relation between the premise and hypothesis:")[0].strip()
                        new_row['explanation'] = row['explanations_story'].split("Real-world example:",1)[1].split("Relation between the premise and hypothesis:")[1].strip()
                else:
                    new_row['premise'] = row['input_story'].split("Premise:")[1].split("Hypothesis:")[0].strip()
                    new_row['hypothesis'] = row['input_story'].split("Hypothesis:")[1].strip()
                    new_row['explanation'] = row['explanations_story'].strip()
            except Exception as e:
                print(e)
                print(row)
                print(idx)
                new_row=eval(input("Modified input:\n"))
                
            new_row['label_story'] = row['label_story']
            new_data.append(new_row)
        self.content = new_data

    def store_data(self, file):
        write_dict_to_csv(self.content, file, verbose=True)

if __name__ == '__main__':
    dm = DataManager()
    dm.generate_story_file()
    for split in ['test', 'dev']:
        post_processor = PostProcessor()
        story_file = dm.story_file_format.format(split)
        post_processor.step1_reformat_answer(story_file)
        post_processor.step2_make_column()
        post_processor.store_data(story_file.replace(".csv", "_post.csv"))