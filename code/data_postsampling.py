import pandas as pd
import os
id = None
def subsample(input_path, output_path):
    test_path = "/cluster/project/sachan/zhiheng/zhiheng/develop_codeversion/corr2cause/data_old/test_new.csv"
    print("Subsampling file: {}".format(input_path))
    # print("Saving to: {}".format(output_path))
    # Load DataFrames
    test_df = pd.read_csv(test_path)
    input_df = pd.read_csv(input_path)
    input_df = input_df.drop_duplicates(subset='prompt', keep='first')
    # print(input_df.head())
    global id
    if type(id) == type(None):
        # Subsample input_df to match test_df
        # Assuming we're matching 'prompt' in input_df to 'input' in test_df
        subsampled_df = input_df[input_df['prompt'].isin(test_df['input'])]

        # Save the subsampled dataframe
        subsampled_df.to_csv(output_path, index=False)
        id = subsampled_df['id']
        # print(id[:10])
    else: # Select by id
        subsampled_df = input_df[input_df['id'].isin(id)]
        subsampled_df.to_csv(output_path, index=False)
    # Print lengths for verification
    # print("Length of test_df: {}".format(len(test_df)))
    # print("Length of subsampled output_df: {}".format(len(subsampled_df)))
    if len(test_df) != len(subsampled_df):
        print("Length of test_df: {}".format(len(test_df)))
        print("Length of subsampled output_df: {}".format(len(subsampled_df)))
        print("##################################################")
    # print("Length of input_df: {}".format(len(input_df)))
    # print("##################################################")
    # len_unique_prompts_input_df = len(input_df['prompt'].unique())
    # len_unique_inputs_test_df = len(test_df['input'].unique())
    # len_unique_prompts_subsampled_df = len(subsampled_df['prompt'].unique())
    # print("Length of unique prompts in input_df: {}".format(len_unique_prompts_input_df))
    # print("Length of unique inputs in test_df: {}".format(len_unique_inputs_test_df))
    # print("Length of unique prompts in subsampled_df: {}".format(len_unique_prompts_subsampled_df))
    # test_df_inputs_set = set(test_df['input'].unique())
    #
    # # Convert the 'prompt' column of subsampled_df to a set
    # subsampled_df_prompts_set = set(subsampled_df['prompt'].unique())
    # # Find prompts that exist in test_df but not in subsampled_df
    # prompts_in_test_not_in_subsampled = test_df_inputs_set - subsampled_df_prompts_set
    # # Output the prompts
    # print(list(prompts_in_test_not_in_subsampled))
    # exit(0)
    return subsampled_df


def iterate_folder(input_dir, output_dir):
    # Iterate through all files in the input directory
    for file in os.listdir(input_dir):
        # Check if the file ends with '_test.csv'
        if file.endswith('_test.csv'):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            # Apply the subsample function
            subsample(input_path, output_path)


if __name__ == '__main__':
    input_dir = "/cluster/project/sachan/zhiheng/zhiheng/develop_codeversion/corr2cause/data/outputs"
    output_dir = "/cluster/project/sachan/zhiheng/zhiheng/develop_codeversion/corr2cause/data_old/outputs"
    iterate_folder(input_dir, output_dir)
