# CausalNLI

This repo contains the code and data for the paper: 

​		**Can Large Language Models Infer Causation from Correlation?** (2023)

​		Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrinmaya Sachan, Rada Mihalcea, Mona Diab\*, Bernhard Schölkopf\*       (*: Co-supervision)

### File Structure

- `code/`:
  - Data-related codes: `data_gen.cpp`, `data_verbalize.py`, `data_stats.py`
  - Model-related codes: `run_model.py` 

- `data/`: Our data folder. Feel free to download the data linked below to this folder.



### Data Download

Feel free to download our files at this [Google Drive link](https://drive.google.com/drive/folders/1a90-cCOFvrtbk30nXaW5GgFY_74A56c0):

- `data_2class` (727M): Our main data. The "entailment" class is the "valid" class (v=1) as mentioned in the paper.
- `data_2class_from_Z` (718M): The variable refactor version which we swap A, B, C, ... to Z, Y, W, ...
- `data_3class` (24M): just as supplementary, we also created the 3-class version of our data, containing "entailment", "neutral" (the hypothesized causal relation can be true sometimes, but there is no sufficient information), and "contradiction" (the hypothesized causal relation does not hold on any causal graphs in the Markov Equivalence class)
- `data_3class_from_Z` (798M) 
- `data_paraph` (4.4M): The paraphrased of our test set (for our robustness test)
- `outputs` (232M): all the output files of our models

### How to Run

#### 1. Generate our data

* Graph Generation, Remove Duplication and Store: `reconstruct_graph` namespace in `data_gen.cpp`
* D-seperation Check: `d_seperate` namespace in `data_gen.cpp
* Node Relationship Generation: `node_relations` namespace in `data_gen.cpp`
* Compose the synthetic dataset(TODO; not in my C++ code)

To reproduce our result, run the following command:

* `g++ data_gen.cpp -o data_gen`
* `./data_gen`

To change the graph size `n`, manually change the constant in the line 11 of the code; to change the output path, change the line 515 in the code. Note that for some larger n, you should enhance the length of array `uniqueDag`, but a high length on some OS will result in compiler error due to the default memory space.

#### 2. Evaluating existing models

Todo: Zhijing

To generate predictions with Alpaca and Llama models specify the location of the input file, output file and the weights in  `code/run_llama.py` and/or `code/run_alpaca.py` and run

```bash
python code/run_llama.py
```

#### 3. Finetuning models

To finetune BERT-based models, check the code in `code/finetune/`. Feel free to customize the code for you to use.

The script below provides an example of finetuning the `roberta-large-mnli` model:

```bash
python3 train.py \
    --dataset_df_dir <dataset_df_dir> \
    --splits_filename train.csv val.csv test.csv \
    --text_col input \
    --y_col label \
    --class_weight automatic \
    --seed 42 \
    --model_save_dir <model_save_dir> \
    --log_dir <log_dir> \
    --iter_time_span 1000 \
    --output_type categorical \
    --num_classes 3 \
    --pretrained_model roberta-large-mnli \
    --lr 1e-5 \
    --max_length 512 \
    --csv_output_path <csv_output_path> \
    --n_epochs 10
```

To evaluate the model, please refer to the script below:

```bash
python3 eval.py \
    --dataset_df_dir <dataset_df_dir> \
    --splits_filename test.csv test.csv test.csv \
    --text_col input \
    --y_col label \
    --seed 42 \
    --model_load_path <model_load_path> \
    --log_dir <log_dir> \
    --pretrained_model roberta-large-mnli \
    --csv_output_path <csv_output_path> \
    --output_type categorical \
    --num_classes 3 \
    --img_output_dir <img_output_dir>
```

### Contact

- For any questions about  `data_gen.cpp`, contact [Zhiheng Lyu](https://cogito233.github.io/)
- For any questions about finetuning models, contact [Jiarui Liu](https://jiarui-liu.github.io/)
- For any other question, contact [Zhijing Jin](https://zhijing-jin.com)
