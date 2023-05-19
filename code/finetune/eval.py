import argparse
import logging
import os
import sys
import pandas as pd
import torch
from datetime import datetime
from models import DfModel as DM
from utils import log_args, add_args, LoggerWritter, modify_dict_from_dataparallel
from analysis import ErrorAnalysis as ALS


def eval_preparation(args):

    if not os.path.exists(f'{args.csv_output_path}'):
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

        dm.dataloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False
        }
        dm.get_dataloader()
        dm.args.dropout_rate_curr = args.dropout_rate
        if dm.args.output_type in ['binary', 'categorical']:
            dm.loss_f = torch.nn.CrossEntropyLoss()
        elif dm.args.output_type == 'real':
            dm.loss_f = torch.nn.MSELoss()

        log.info('Get model..')
        dm.get_model()

        log.info('Get model checkpoint..')
        # if loading a state_dict saved without removing extra keys in the dict,
        dm.model.load_state_dict(modify_dict_from_dataparallel(
            torch.load(args.model_load_path), args))
        # dm.model.load_state_dict(torch.load(args.model_load_path))
        dm.model.eval()

        return dm
    else:
        raise NotImplementedError

    # log.info('Begin error analysis:')
    # als = ALS(args)
    # als.get_classification_report()
    # als.get_classification_heatmap()


def eval(dm, args):
    log.info('Begin evaluation:')
    dm.eval(split=args.eval_on_data, store_csv=True, report_analysis=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # evaluation
    # dataset related
    parser.add_argument('--dataset_df_path', type=str,
                        default=None, help='dataset dataframe input path')
    parser.add_argument('--dataset_split', nargs='*', type=float, default=[
                        0.8, 0.1], help='define dataset training and validation splits proportions')
    # (2):
    parser.add_argument('--dataset_df_dir', type=str,
                        default=None, help='dataset dataframe splits input dir')
    parser.add_argument('--splits_filename', nargs='*',
                        type=str, default=['train.csv', 'val.csv', 'test.csv'])

    parser.add_argument('--text_col', type=str, default='text')
    parser.add_argument('--y_col', type=str, default='y')
    parser.add_argument('--num_numeric_features', type=int, default=0)
    parser.add_argument('--numeric_features_col',
                        nargs='*', type=str, default=None)
    parser.add_argument('--eval_on_data', type=str, default='test',
                        choices=['train', 'val', 'test'])

    # macro settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='flexible',
                        choices=['flexible', 'cpu'])
    parser.add_argument('--model_load_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--csv_output_path', type=str, default=None)  # .csv
    parser.add_argument('--dataparallel', type=bool, default=True)

    # training task related
    parser.add_argument('--output_type', type=str, default='binary',
                        choices=['binary', 'categorical', 'real'])
    # if output_type is categorical
    parser.add_argument('--num_classes', type=int, default=None)

    # pretrained model related
    parser.add_argument('--max_length', type=int, default=512,
                        help='the input length for bert')
    parser.add_argument('--pretrained_model', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased', 'allenai/scibert_scivocab_uncased', 'allenai/longformer-base-4096',
                                 'microsoft/deberta-v3-large', 'roberta-large-mnli', 'textattack/bert-base-uncased-MNLI', 'madlag/bert-large-uncased-mnli'])

    # hyperparams below will have accompanying current value as attributes in args.
    # For example, <batch_size> will have accompanying value <batch_size_curr> in the actual training.
    # list for searching hyperms
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    # list for searching hyperms, a list of list
    parser.add_argument('--hidden_dim_curr', nargs='*', type=int, default=None)

    # error analysis
    parser.add_argument('--img_output_dir', type=str, default=None)
    # regression
    # parser.add_argument('--')

    args = parser.parse_args()
    args.dataset_class_dir = None

    # logging
    now = datetime.now()
    now = now.strftime("%Y%m%d-%H:%M:%S")
    args.now = now
    handler = logging.FileHandler(filename=f'{args.log_dir}eval-{now}.log')
    log = logging.getLogger('bert_tune')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    sys.stderr = LoggerWritter(log.warning)

    log_args(args, log)
    args = add_args(args)
    dm = eval_preparation(args)
    eval(dm, args)
