import argparse
import logging
import sys
import pandas as pd
from datetime import datetime
from models import DfModel as DM
from utils import log_args, add_args, LoggerWritter


def train(args):

    log.info('Initialize DfModel class and get dataset:')
    if args.dataset_df_path is None:
        dm = DM(args, get_df=False)
        log.info(
            f'Training set path: {args.dataset_df_dir}{args.splits_filename[0]}')
        dm.get_df(
            path_train=f'{args.dataset_df_dir}{args.splits_filename[0]}',
            path_val=f'{args.dataset_df_dir}{args.splits_filename[1]}',
            path_test=f'{args.dataset_df_dir}{args.splits_filename[2]}'
        )
    elif args.dataset_df_dir is None:
        dm = DM(args)
        dm.get_df_split(split_portion=args.dataset_split)

    log.info('Get dataloader and tokenizer:')
    dm.set_device()
    dm.get_tokenizer()

    log.info('Begin grid search:')
    dm.grid_search()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset related
    # chooce one of the two options below:
    # (1):
    parser.add_argument('--dataset_df_path', type=str,
                        default=None, help='dataset dataframe input path')
    parser.add_argument('--dataset_split', nargs='*', type=float, default=[
                        0.8, 0.1], help='define dataset training and validation splits proportions')
    # (2):
    parser.add_argument('--dataset_df_dir', type=str,
                        default=None, help='dataset dataframe splits input dir')
    parser.add_argument('--splits_filename', nargs='*',
                        type=str, default=['train.csv', 'val.csv', 'test.csv'])
    # dataset customization
    parser.add_argument('--dataset_class_dir', type=str, default=None,
                        help='a path to the dataset class that wants to use')
    parser.add_argument('--dataset_class_name', type=str, default='Dataset')

    parser.add_argument('--text_col', type=str, default='text')
    parser.add_argument('--y_col', type=str, default='y')
    parser.add_argument('--num_numeric_features', type=int, default=0)
    parser.add_argument('--numeric_features_col',
                        nargs='*', type=str, default=None)
    parser.add_argument('--class_weight', type=str, default=None,
                        help='"automatic" or written as the strong format of a list, e.g., "1,10"')

    # macro settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='flexible',
                        choices=['flexible', 'cpu'])
    parser.add_argument('--model_save_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--csv_output_path', type=str, default=None)  # .csv

    # load checkpoint
    parser.add_argument('--model_load_path', type=str, default=None)

    # training task related
    parser.add_argument('--output_type', type=str, default='binary',
                        choices=['binary', 'categorical', 'real'])
    # if output_type is categorical
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--iter_time_span', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'rmsprop', 'sgd'])
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    # if optimizer == adamw:
    parser.add_argument('--adamw_epsilon', type=float, default=1e-8)
    parser.add_argument('--adamw_warmup_steps', type=int, default=0)

    # pretrained model related
    parser.add_argument('--max_length', type=int, default=512,
                        help='the input length for bert')
    parser.add_argument('--pretrained_model', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_uncased', 'allenai/longformer-base-4096',
                                 'microsoft/deberta-v3-large', 'roberta-large-mnli', 'textattack/bert-base-uncased-MNLI', 'madlag/bert-large-uncased-mnli', 'textattack/roberta-base-MNLI', 'sileod/roberta-base-mnli'])

    # training hyperparameters related
    parser.add_argument('--n_epochs', type=int, default=5)
    # hyperparams below will have accompanying current value as attributes in args.
    # For example, <batch_size> will have accompanying value <batch_size_curr> in the actual training.
    parser.add_argument('--batch_size', nargs='*', type=int,
                        default=[8])  # list for searching hyperms
    parser.add_argument('--dropout_rate', nargs='*', type=float,
                        default=[0.1])  # list for searching hyperms
    parser.add_argument('--lr', nargs='*', type=float,
                        default=[1e-5])  # list for searching hyperms
    # list for searching hyperms, a list of list
    parser.add_argument('--hidden_dim_curr', nargs='*', type=int, default=None)

    args = parser.parse_args()

    # logging
    now = datetime.now()
    now = now.strftime("%Y%m%d-%H:%M:%S")
    args.now = now
    handler = logging.FileHandler(filename=f'{args.log_dir}train-{now}.log')
    log = logging.getLogger('bert_tune')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    sys.stderr = LoggerWritter(log.warning)

    log_args(args, log)
    args = add_args(args)
    train(args)
