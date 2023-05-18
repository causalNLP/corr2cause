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



### Contact

- For any questions about  `data_gen.cpp`, contact [Zhiheng Lyu](https://cogito233.github.io/)
- For any questions about finetuning models, contact [Jiarui Liu](https://jiarui-liu.github.io/)
- For any other question, contact [Zhijing Jin](https://zhijing-jin.com)