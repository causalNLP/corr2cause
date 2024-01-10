# Corr2Cause Project

This repo contains the code and data for the paper: 

​		[**Can Large Language Models Infer Causation from Correlation?**](http://arxiv.org/abs/2306.05836) (2023)

​		*Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrinmaya Sachan, Rada Mihalcea, Mona Diab\*, Bernhard Schölkopf\**       (*: Co-supervision)


### File Structure

- `code/`:
  - Data-related codes: `data_gen.cpp`, `data_verbalize.py`, `data_stats.py`
  - Model-related codes: `run_*.py` 

- `data/`: Our data folder. Feel free to download the data from https://huggingface.co/datasets/causalnlp/corr2cause to this folder.

### How to Run

#### Step 1. Corr2Cause Data Generation

**Shortcut:** You can directly download the data from https://huggingface.co/datasets/causalnlp/corr2cause

**To generate the data yourself:** 

To reproduce our result, run the following command:

* `g++ data_gen.cpp -o data_gen`
* `./data_gen`

To change the graph size `n`, manually change the constant in the line 11 of the code; to change the output path, change the line 515 in the code. Note that for some larger n, you should enhance the length of array `uniqueDag`, but a high length on some OS will result in compiler error due to the default memory space.

What each file does:

* Graph generation, remove duplication and store: `reconstruct_graph` namespace in `data_gen.cpp`
* D-seperation check: `d_seperate` namespace in `data_gen.cpp`
* Node relationship generation: `node_relations` namespace in `data_gen.cpp`
* Compose the synthetic dataset: TBA

#### Step 2. Evaluating Existing Models

**Shortcut:** To replicate the results, since we save all the output files of our models into the `output` folder (232M). Feel free to download them from [here](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.VYGWHY), and use the `data_v2/outputs/` folder by `mv corr2cause_supplementary/data_v2/outputs/ ./data/`. Then you can run:

```bash
python code/run_model.py -model_types random gpt huggingface coauthor_files
```

**To generate the predictions yourself:** 

To generate predictions with Alpaca and Llama models specify the location of the input file, output file and the weights in  `code/run_llama.py` and/or `code/run_alpaca.py` and run

```bash
python code/run_llama.py
```

To generate predictions for other models, run

```bash
python code/run_model.py -inference_mode -model_types random gpt huggingface coauthor_files
```

#### Step 3. Finetuning Models

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



### Other Data

If you need extra info, we open-source all different formats of the data and outputs [here](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.VYGWHY). Feel free to use the data_v2 folder by `mv corr2cause_supplementary/data_v2/ ./data/`. Inside this folder, you can see the following subfolders:

- `outputs` (232M): all the output files of our models
- `data_2class` (727M): Our main data. The "entailment" class is the "valid" class (v=1) as mentioned in the paper.
- `data_2class_from_Z` (718M): The variable refactor version which we swap A, B, C, ... to Z, Y, W, ...
- `data_3class` (24M): just as supplementary, we also created the 3-class version of our data, containing "entailment", "neutral" (the hypothesized causal relation can be true sometimes, but there is no sufficient information), and "contradiction" (the hypothesized causal relation does not hold on any causal graphs in the Markov Equivalence class)
- `data_3class_from_Z` (798M) 
- `data_paraph` (4.4M): The paraphrased of our test set (for our robustness test)

### Contact

- For any questions about the data generation process, contact [Zhiheng Lyu](https://cogito233.github.io/)
- For any questions about finetuning models, contact [Jiarui Liu](https://jiarui-liu.github.io/)
- For any other questions, contact [Zhijing Jin](https://zhijing-jin.com)
