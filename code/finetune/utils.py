import sys


def plot_heatmap(df, vmin=None, vmax=None, filepath='../heatmap.pdf'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=True, vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax)
    plt.savefig(filepath, dpi=600)


def RMSELoss(yhat, y):
    import torch
    return torch.sqrt(torch.mean((yhat-y)**2))

# logger funcs


def log_args(args, log):
    for argname, argval in vars(args).items():
        log.info(f'{argname.replace("_"," ").capitalize()}: {argval}')


def add_args(args):
    import torch
    # check model type and add feature_size
    if args.pretrained_model in ['bert-base-cased',
                                 'bert-base-uncased',
                                 'roberta-base',
                                 'textattack/bert-base-uncased-MNLI',
                                'textattack/roberta-base-MNLI',
                                'sileod/roberta-base-mnli',
                                 'allenai/scibert_scivocab_uncased',
                                 'allenai/longformer-base-4096']:
        args.feature_size = 768
    elif args.pretrained_model in ['bert-large-cased',
                                   'bert-large-uncased',
                                   'madlag/bert-large-uncased-mnli',
                                   'roberta-large',
                                   'roberta-large-mnli',
                                   'microsoft/deberta-v3-large']:
        args.feature_size = 1024

    # dataparallel
    args.dataparallel = False
    if torch.cuda.device_count() > 1:
        args.dataparallel = True
    return args


class LoggerWritter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

# modify state dict from data parallel


def modify_dict_from_dataparallel(state_dict, args):
    if args.dataparallel:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
        # if roberta-large-mnli
        # if k == "classifier.dense.weight":
        #     new_state_dict['bert_model.pooler.dense.weight'] = v
        # elif k == "classifier.dense.bias":
        #     new_state_dict['bert_model.pooler.dense.bias'] = v
        # elif k == "classifier.out_proj.weight":
        #     new_state_dict['l1.weight'] = v
        # elif k == "classifier.out_proj.bias":
        #     new_state_dict['l1.bias'] = v
        # else:
        #     name = 'bert_model' + k[7:]  # remove `module.`
        #     new_state_dict[name] = v

        # if bert-base-mnli
        # if k == 'classifier.weight':
        #     new_state_dict['l1.weight'] = v
        # elif k == 'classifier.bias':
        #     new_state_dict['l1.bias'] = v
        # else:
        #     name = 'bert_model' + k[4:]  # remove `module.`
        #     new_state_dict[name] = v

        # if longformer and others
        # new_state_dict = state_dict
        # break

        # if bert-base
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict
