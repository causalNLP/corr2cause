import logging
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

sys.path.append('../')
from utils import LoggerWritter

handler = logging.StreamHandler(sys.stdout)
log = logging.getLogger('analysis')
log.addHandler(handler)
log.setLevel(logging.INFO)
sys.stderr = LoggerWritter(log.warning)


class ErrorAnalysis():
    """Class for analyzing model evaluation results.

    Parameters:
    -----------
    args: a python class with required attributes.

        1. common

            csv_input_path: str, path to the csv result file.
            pred_col: str, name of the prediction column in the csv file.
            truth_col: str, name of the ground truth column in the csv file.

        2. loss
            loss_type: str, call the loss function in `torch.nn.functional`.

        3. report
            task_type: str, in `['binary', 'categorical', 'real']`.

        4. image
            img_save_dir: str, directory to save images.

        5. save to json
            json_output_path: str, path to the output json file.

    """

    def __init__(self, args):
        self.args = args

        # get df
        # data_index, y_truth, y_pred
        log.info('Read csv file..')
        self.df = pd.read_csv(f'{self.args.csv_input_path}')

    # loss

    @staticmethod
    def calc_loss(df, truth_col, pred_col, loss_type, **kwargs):
        if loss_type == 'r2_score':
            from sklearn.metrics import r2_score
            return r2_score(df[truth_col], df[pred_col])
        elif loss_type == 'rmse_loss':
            import torch
            import torch.nn.functional as F
            return torch.sqrt(F.mse_loss(torch.tensor(df[pred_col]), torch.tensor(df[truth_col])))
        elif loss_type in ['mse_loss', 'cross_entropy']:
            import torch.nn.functional as F
            loss_func = getattr(F, loss_type)
            return loss_func(
                input=df[pred_col],
                target=df[truth_col],
                **kwargs
            )

    def get_calc_loss(self):
        log.info(f'Calculate {self.args.loss_type} loss(es):')
        loss_dict = {}
        for idx, loss_str in enumerate(self.args.loss_type):
            loss_dict[loss_str] = ErrorAnalysis.calc_loss(
                self.df, self.args.truth_col, self.args.pred_col, loss_type=loss_str, **self.args.loss_kwargs[idx])
        return loss_dict

    # classification

    @staticmethod
    def clf_report(df, truth_col, pred_col):
        return classification_report(
            df[truth_col].astype(int).values.tolist(), df[pred_col].values.tolist(), digits=4)

    def get_clf_report(self):
        # only for classification
        log.info(f'Classification report:')
        report = ErrorAnalysis.clf_report(
            self.df, self.args.truth_col, self.args.pred_col)
        log.info(report)
        return report

    @staticmethod
    def clf_heatmap(df, truth_col, pred_col):
        """
        I.e., confusion matrix.
        """
        import seaborn as sns
        import numpy as np
        from matplotlib.colors import LogNorm
        import matplotlib.pyplot as plt
        # get confusion matrix a
        # sklearn.metrics.confusion_matrix
        a = np.zeros((int(df[truth_col].max()+1),
                      int(df[pred_col].max()+1)))
        for _, row in df.iterrows():
            a[row[truth_col], row[pred_col]] += 1
        fig, ax = plt.subplots()
        cmap = sns.cubehelix_palette(light=0.7, as_cmap=True)
        s = sns.heatmap(data=a, fmt='.0f', norm=LogNorm(),
                        annot=True, cmap=cmap)
        # s = sns.heatmap(data=a, norm=LogNorm(), annot=True,
        # fmt='.0f', cmap='rocket_r')
        s.set(ylabel=truth_col, xlabel=pred_col)
        return fig

    def get_clf_heatmap(self):
        # only for classification
        log.info(f'Classification heatmap:')
        import matplotlib.pyplot as plt
        fig = ErrorAnalysis.clf_heatmap(
            self.df, self.args.truth_col, self.args.pred_col)
        # plt.show()
        output_path = f'{self.args.img_save_dir}classification_heatmap.pdf'
        plt.savefig(
            output_path, bbox_inches='tight')
        log.info(
            f'Heatmap saved to {output_path}')
        return output_path

    @staticmethod
    def factorize_df(df, truth_col, pred_col):
        # only for regression
        clfish_df = df.copy()
        clfish_df[truth_col] = pd.factorize(
            df[truth_col], sort=True)[0]
        clfish_df[pred_col] = pd.factorize(
            df[pred_col], sort=True)[0]
        return clfish_df

    # regression

    @staticmethod
    def get_clfish_df(in_df, truth_col, pred_col, thres=20, n_blocks=5):
        def get_interv_list(df, col):
            col_min, col_max = df[col].max(), df[col].min()
            chunk_len = (col_max - col_min) / n_blocks
            interv_list = [int(item) for item in (
                (df[col] - col_min) // chunk_len).tolist()]
            return interv_list

        truth_set, pred_set = set(), set()
        df = in_df.copy()
        round_to_int = True
        truth_list, pred_list = [], []
        for _, row in df.iterrows():
            if len(truth_set) > thres or len(pred_set) > thres:
                round_to_int = False
                # del truth_list, pred_list
            else:
                truth_set.add(round(row[truth_col]))
                pred_set.add(round(row[pred_col]))
                truth_list.append(round(row[truth_col]))
                pred_list.append(round(row[pred_col]))

        if round_to_int:
            log.info('Round to integer..')
            df[truth_col] = truth_list
            df[pred_col] = pred_list
        else:
            log.info('Aggregate into chunks..')
            df[truth_col] = get_interv_list(df, truth_col)
            df[pred_col] = get_interv_list(df, pred_col)

        return df

    @staticmethod
    def reg_report(df, truth_col, pred_col, **kwargs):
        new_df = ErrorAnalysis.get_clfish_df(
            df, truth_col, pred_col, **kwargs)
        report = ErrorAnalysis.clf_report(
            new_df, truth_col, pred_col
        )
        return report

    def get_reg_report(self, **kwargs):
        """
        kwargs: thres (default=20), n_blocks (default=5)
        """
        # only for regression
        log.info('Regression report:')
        report = ErrorAnalysis.reg_report(
            self.df, self.args.truth_col, self.args.pred_col, **kwargs)
        log.info(report)
        return report

    @staticmethod
    def reg_heatmap(df, truth_col, pred_col):
        import seaborn as sns
        from matplotlib.colors import LogNorm
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        s = sns.jointplot(df[truth_col], df[pred_col],
                          kind='kde', x=truth_col, y=pred_col, fill=True, cmap='Blues')
        s.set(xlabel=truth_col, ylabel=pred_col)
        plt.tight_layout()
        return fig

    def get_reg_heatmap(self):
        # only for regression
        # jointplot
        import matplotlib.pyplot as plt
        fig = ErrorAnalysis.reg_heatmap(
            self.df, self.args.truth_col, self.args.pred_col)
        output_path = f'{self.args.img_save_dir}regression_heatmap.pdf'
        plt.savefig(output_path, bbox_inches='tight')
        log.info(
            f'Heatmap saved to {output_path}')
        return output_path

    @staticmethod
    def get_histogram(col1, col2, col1_name=None, col2_name=None, df=None, **kwargs):
        """
        df
            - if None, then col1 and col2 should be two lists or series from two different pandas dataframes.
            - if not None, then col1 and col2 should be two column names of df.
        """
        import matplotlib.pyplot as plt
        if kwargs == {}:
            kwargs = {
                'alpha': 0.65,
                'grid': True,
                'bins': 10,
                'density': True,
            }
        if df is None:
            fig, ax = plt.subplots()
            plt.hist(col1, bins=kwargs['bins'],
                     alpha=kwargs['alpha'], label=col1_name)
            plt.hist(col2, bins=kwargs['bins'],
                     alpha=kwargs['alpha'], label=col2_name)
            # plt.xlabel()
            plt.xlabel("Uncertainty", size=14)
            plt.ylabel("Count", size=14)
            # plt.title("Multiple Histograms with Matplotlib")
            plt.legend()
        else:
            ax = df[[col1, col2]].plot(
                kind='hist',
                **kwargs
            )
            fig = ax.get_figure()
        return fig

    # save to json

    def save_to_json(self):
        log.info('Save to json..')
        output_dict = {}
        if self.args.task_type != 'real':
            output_dict['report'] = self.get_clf_report()
            output_dict['loss'] = self.get_calc_loss()
            output_dict['heatmap'] = self.get_clf_heatmap()

        else:
            output_dict['report'] = self.get_reg_report(
                **self.args.reg_heatmap_kwargs)
            output_dict['loss'] = self.get_calc_loss()
            output_dict['heatmap'] = self.get_reg_heatmap()

        with open(self.args.json_output_path, 'w') as file:
            json.dump(output_dict, file)

    # correlation

    @staticmethod
    def get_corr(df, col1, col2):
        return df[col1].corr(df[col2])

    @staticmethod
    def get_corr_heatmap(df, col1, col2):
        return ErrorAnalysis.clf_heatmap(df, col1, col2)

    # contingency table: binary or categorical
    @staticmethod
    def get_crosstab(df, col1, col2):
        return pd.crosstab(df[col1], df[col2])

    # matthews corrcoef: binary
    @staticmethod
    def get_matthews_coef(df, col1, col2):
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(df[col1], df[col2])

    # hypothesis test
    # https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce
    # maybe make it a new class?

class ModelInterpretAnalysis():
    def __init__(self, dm, args):
        import shap
        import transformers
        import nlp
        import torch
        import numpy as np
        import scipy as sp
        self.dm = dm
        self.args = args
    
    # def 
    