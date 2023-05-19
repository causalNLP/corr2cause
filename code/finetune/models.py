import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import RMSELoss, modify_dict_from_dataparallel
from torchmetrics import R2Score
from sklearn import metrics

import logging
log = logging.getLogger('bert_tune')


def get_tokenizer(pretrained_model):
    if pretrained_model in ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased', 'allenai/scibert_scivocab_uncased']:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model)
    elif pretrained_model in ['roberta-base', 'roberta-large']:
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model)
    elif pretrained_model == 'allenai/longformer-base-4096':
        from transformers import LongformerTokenizer
        tokenizer = LongformerTokenizer.from_pretrained(pretrained_model)
    elif pretrained_model in ['microsoft/deberta-v3-large', 'roberta-large-mnli', 'textattack/bert-base-uncased-MNLI', 'madlag/bert-large-uncased-mnli', 'sileod/roberta-base-mnli']:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    return tokenizer


def get_model(pretrained_model, clf, num_labels=None):
    if clf in ['binary', 'categorical']:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model)
    else:
        if pretrained_model in ['bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased', 'allenai/scibert_scivocab_uncased']:
            from transformers import BertModel
            model = BertModel.from_pretrained(
                pretrained_model)
        elif pretrained_model in ['roberta-base', 'roberta-large']:
            from transformers import RobertaModel
            model = RobertaModel.from_pretrained(pretrained_model)
        elif pretrained_model == 'allenai/longformer-base-4096':
            from transformers import LongformerModel
            model = LongformerModel.from_pretrained(pretrained_model)
        elif pretrained_model in ['microsoft/deberta-v3-large', 'roberta-large-mnli', 'textattack/bert-base-uncased-MNLI', 'madlag/bert-large-uncased-mnli', 'sileod/roberta-base-mnli']:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(pretrained_model,
                                              num_labels=num_labels,
                                              output_attentions=False,
                                              output_hidden_states=False,)
    return model


def _get_module_list(args, output_dim=2):
    modules = []
    for idx, hidden_dim in enumerate(args.hidden_dim_curr):
        if idx == 0:
            # print(args.feature_size, type(args.feature_size))
            # print(args.num_numeric_features, type(args.num_numeric_features))
            # print(hidden_dim_curr)
            modules.append(torch.nn.Linear(args.feature_size +
                                           args.num_numeric_features, hidden_dim))
            # modules.append(torch.nn.ReLU())
        else:
            modules.append(torch.nn.Linear(
                args.hidden_dim_curr[idx-1], hidden_dim))
            # modules.append(torch.nn.ReLU())

    modules.append(torch.nn.Linear(args.hidden_dim_curr[-1], output_dim))
    return torch.nn.Sequential(*modules)


def read_optimizer(args, model, len_train_loader):
    args.optimizer_func, args.scheduler_func = None, None
    if args.optimizer == 'adamw':
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        args.optimizer_func = AdamW(optimizer_grouped_parameters,
                                    lr=args.lr_curr,
                                    eps=args.adamw_epsilon)
        args.scheduler_func = get_linear_schedule_with_warmup(
            args.optimizer_func,
            num_warmup_steps=args.adamw_warmup_steps,
            num_training_steps=len_train_loader * args.n_epochs
        )
    elif args.optimizer == 'adam':
        args.optimizer_func = torch.optim.Adam(
            model.parameters(), lr=args.lr_curr
        )
    elif args.optimizer == 'rmsprop':
        args.optimizer_func = torch.optim.RMSprop(
            params=model.parameters(),
            lr=args.lr_curr
        )
    elif args.optimizer == 'sgd':
        args.optimizer_func = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_curr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            # weight_decay=0.2, #l2 regularization
        )
        args.scheduler_func = torch.optim.lr_scheduler.ReduceLROnPlateau(
            args.optimizer_func,
            "min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08,
        )
    else:
        raise NotImplementedError
    return args


def get_class_weight(weight_input, class_dict):
    if weight_input == 'automatic':
        weight = torch.tensor([1/item[1] for item in class_dict])
    else:
        weight = torch.tensor([float(item)
                               for item in weight_input.split(',')])
    log.info('Class weights:')
    log.info(weight)
    return weight


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.args = args
        self.texts = data[args.text_col].to_list()  # text
        if args.num_numeric_features > 0:
            for col in args.numeric_features_col:
                setattr(self, f'numeric_{col}', data[col].to_list())

        self.y = data[args.y_col].to_list()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])

        y = self.y[index]

        if self.args.num_numeric_features > 0:
            num_features = []
            for col in self.args.numeric_features_col:
                attrib = getattr(self, f'numeric_{col}')
                num_features.append(attrib[index])
            return text, y, index, num_features

        return text, y, index


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.bert_model = get_model(
            args.pretrained_model, args.output_type, args.num_classes)
        self.drop = torch.nn.Dropout(p=args.dropout_rate_curr)

        if args.pretrained_model == 'allenai/scibert_scivocab_uncased':
            for name, param in self.bert_model.named_parameters():
                # specify which layer to learn
                if not ('model.pooler' in name or 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'encoder.layer.9' in name):
                    param.requires_grad = False

        if args.hidden_dim_curr is None:
            self.l1 = torch.nn.Linear(
                args.feature_size+args.num_numeric_features, 1)
            # args.feature_size+args.num_numeric_features, args.num_classes)
        else:
            self.l1 = _get_module_list(args, output_dim=1)
            # self.l1 = _get_module_list(args, output_dim=args.num_classes)

    def forward(self, tokenized_text, numeric_features=None):
        if self.args.pretrained_model != 'microsoft/deberta-v3-large':
            text_rep = self.drop(self.bert_model(tokenized_text).pooler_output)
        else:
            text_rep = self.drop(self.bert_model(
                tokenized_text)[-1])
        # log.info(text_rep.shape)
        if numeric_features is not None:
            text_rep = torch.cat((text_rep, numeric_features), dim=1)

        # out = F.relu(self.l1(text_rep))
        out = self.l1(text_rep)
        # log.info(out.shape)
        # out = self.l2(out)

        return out


class DfModel():
    def __init__(self, args, get_df=True):
        log.info('DfModel class initialization')
        log.info(locals())
        self.args = args
        self.set_seed(args.seed)
        if get_df:
            self.get_df(args.dataset_df_path)

    def set_seed(self, seed=42):
        log.info('DfModel set_seed..')
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def get_df(self, path=None, path_train=None, path_val=None, path_test=None):
        log.info('DfModel get_df..')
        if path is not None:
            # load all data and then call get_df_split to obtain data splits
            self.df = pd.read_csv(
                path, engine='python')
            # self.get_df_split()
        else:
            # load splited data
            self.df_train = pd.read_csv(path_train, engine='python')
            self.df_val = pd.read_csv(path_val, engine='python')
            self.df_test = pd.read_csv(path_test, engine='python')

    def get_df_split(self, split_portion=[0.8, 0.1]):
        log.info('DfModel get_df_split..')
        assert sum(split_portion) <= 1
        self.df_train, self.df_test = train_test_split(
            self.df, test_size=1-split_portion[0], random_state=self.seed)
        self.df_val, self.df_test = train_test_split(
            self.df_test, test_size=(1-split_portion[1]-split_portion[0])/(1-split_portion[0]), random_state=self.seed)
        del self.df

    def set_device(self):
        log.info('DfModel set_device..')
        if self.args.device == 'flexible':
            # flexible choose
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        elif self.args.device == 'cpu':
            # only cpu
            use_cuda = False
            self.device = torch.device('cpu')

    def get_dataloader(self):
        log.info('DfModel get_dataloader..')
        if self.args.dataset_class_dir is None:
            DatasetClass = Dataset
        else:
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location(
                "module.name", self.args.dataset_class_dir)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = module
            spec.loader.exec_module(module)
            DatasetClass = getattr(module, self.args.dataset_class_name)

        self.train_data = DatasetClass(self.df_train, self.args)
        self.val_data = DatasetClass(self.df_val, self.args)
        self.test_data = DatasetClass(self.df_test, self.args)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, **self.dataloader_params)

        # help to store csv file in order
        dataloader_params = self.dataloader_params
        dataloader_params['shuffle'] = False

        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, **dataloader_params)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, **dataloader_params)

    def get_model(self):
        log.info('DfModel get_model..')
        if self.args.output_type in ['binary', 'categorical']:
            self.model = get_model(
                self.args.pretrained_model, self.args.output_type, self.args.num_classes)
        else:
            self.model = Model(self.args)

    def get_tokenizer(self):
        log.info('DfModel get_tokenizer..')
        self.tokenizer = get_tokenizer(self.args.pretrained_model)

    def eval(self, split='val', store_csv=True, report_analysis=True):
        log.info('DfModel eval..')
        if split == 'val':
            dataloader = self.val_loader
        elif split == 'test':
            dataloader = self.test_loader
        r2score = R2Score()
        loss_list = []
        correct_predictions = 0
        predictions = []
        truth_labels = []

        self.model.to(self.device)
        # inspect store_csv
        if store_csv:
            if self.args.numeric_features_col is None:
                output_df = pd.DataFrame(
                    columns=['data_index', 'y_truth', 'y_pred', self.args.text_col])
            else:
                output_df = pd.DataFrame(columns=[
                                         'data_index', 'y_truth', 'y_pred', self.args.text_col, *self.args.numeric_features_col])

        cnt_batch = 0
        with torch.set_grad_enabled(False):
            for bundle in dataloader:
                texts = list(bundle[0])
                label = bundle[1]
                index = bundle[2]
                numeric = bundle[3] if self.args.num_numeric_features > 0 else None
                if numeric is not None:
                    numeric = torch.stack(numeric).permute(
                        1, 0).float().to(self.device)
                model_output = self.model(
                    self.tokenizer(texts, padding=True, return_tensors='pt', truncation=True, max_length=self.args.max_length).input_ids.to(self.device), numeric)

                if self.args.output_type in ['binary', 'categorical']:
                    model_output = model_output.logits

                if self.args.output_type == 'binary' or self.args.output_type == 'categorical':
                    label = label.long()
                    loss = self.loss_f(model_output.cpu(), label)
                    loss_list.append(loss.item())
                    preds = torch.argmax(model_output, dim=1)
                    correct_predictions += torch.sum(preds.cpu() == label)
                    predictions.extend(preds.tolist())
                    truth_labels.extend(label.tolist())

                else:
                    model_output = model_output
                    label = label.float().reshape(-1, 1)
                    loss = self.loss_f(model_output.cpu(), label)
                    loss_list.append(loss.item())
                    preds = model_output.cpu()
                    predictions.extend(preds.tolist())
                    truth_labels.extend(label.tolist())

                if store_csv:
                    # log.info(f'label shape {label.shape}, bundle shape {len(bundle)} {len(bundle[0])}, index shape {index.shape}, pred shape {preds.shape}')
                    for idx in range(len(index)):
                        output_dict = {}
                        output_dict['data_index'] = index[idx].item()
                        if self.args.output_type == 'binary' or self.args.output_type == 'categorical':
                            output_dict['y_truth'] = label[idx].item()
                            output_dict['y_pred'] = preds[idx].item()
                        else:
                            output_dict['y_truth'] = label[idx, 0].item()
                            output_dict['y_pred'] = preds[idx, 0].item()
                        output_dict[self.args.text_col] = texts[idx]
                        if self.args.numeric_features_col is not None:
                            for feature_idx, feature in enumerate(self.args.numeric_features_col):
                                output_dict[feature] = numeric[idx,
                                                               feature_idx].item()
                        output_df = output_df.append(
                            output_dict, ignore_index=True)

                    if cnt_batch % 1000 == 0:
                        output_df.to_csv(
                            f'{self.args.csv_output_path}'.replace('.csv', '_tmp.csv'))
                    # with open(f'{output_dir}eval_output.csv')

                cnt_batch += 1

        if store_csv:
            output_df.to_csv(f'{self.args.csv_output_path}')

        if report_analysis:
            if self.args.output_type == 'binary' or self.args.output_type == 'categorical':
                report = classification_report(
                    truth_labels, predictions, digits=4)
                log.info(f'{split} classification report:')
                log.info(report)
            else:
                rmse_loss = RMSELoss(torch.tensor(predictions),
                                     torch.tensor(truth_labels))
                report = [
                    f'{split} RMSE loss: {rmse_loss}\nR2Score: {r2score(torch.tensor(predictions), torch.tensor(truth_labels))}']
                log.info(report)
            return report, loss_list, correct_predictions

    def train(self,
              epochs, params_str):
        log.info('DfModel train..')

        history = {'train_report': [], 'train_loss': [],
                   'val_report': [], 'val_loss': []}
        if self.args.output_type != 'real':
            best_acc = 0
        else:
            best_acc = 100
        best_history = history.copy()
        train_loss_all_list = []
        r2score = R2Score()

        if self.args.dataparallel:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
        else:
            log.info(f"Let's use {torch.cuda.device_count()} GPU..")
            self.model.to(self.device)

        for epoch in range(epochs):
            log.info(f'Epoch {epoch+1}/{epochs}')
            log.info('-' * 10)
            train_loss_list = []

            self.model.train()
            correct_predictions = 0
            predictions = []
            truth_labels = []
            iter = 0

            with torch.set_grad_enabled(True):
                for bundle in self.train_loader:
                    texts = list(bundle[0])
                    label = bundle[1]
                    index = bundle[2]
                    numeric = bundle[3] if self.args.num_numeric_features > 0 else None
                    self.optimizer.zero_grad()
                    iter += 1
                    if numeric is not None:
                        # print(numeric)
                        numeric = torch.stack(numeric).permute(
                            1, 0).float().to(self.device)
                    input_tokens = self.tokenizer(
                        texts, padding=True, return_tensors='pt', truncation=True, max_length=self.args.max_length).input_ids.to(self.device)

                    if self.args.output_type in ['binary', 'categorical']:
                        model_output = self.model(
                            input_tokens, numeric).logits
                        # print(model_output)
                    else:
                        model_output = self.model(input_tokens, numeric)

                    if self.args.output_type == 'binary' or self.args.output_type == 'categorical':
                        label = label.long()
                        loss = self.loss_f(model_output.cpu(), label)
                        train_loss_list.append(loss.item())
                        preds = torch.argmax(model_output, dim=1)
                        correct_predictions += torch.sum(preds.cpu() == label)
                        predictions.extend(preds.tolist())
                        truth_labels.extend(label.tolist())

                    else:
                        model_output = model_output
                        label = label.float().reshape(-1, 1)
                        loss = self.loss_f(model_output.cpu(), label)
                        # loss.requires_grad = True
                        train_loss_list.append(loss.item())
                        preds = model_output.cpu()
                        predictions.extend(preds.tolist())
                        truth_labels.extend(label.tolist())
        #                 log.info(model_output, label)
                    self.tmp_label = truth_labels
                    self.tmp_pred = predictions
                    if iter % self.args.iter_time_span == 0:
                        if self.args.output_type == 'binary' or self.args.output_type == 'categorical':
                            log.info(
                                f'Iteration {iter}: Training accuracy {metrics.accuracy_score(truth_labels[-self.args.iter_time_span*self.args.batch_size_curr:], predictions[-self.args.iter_time_span*self.args.batch_size_curr:])}, Training loss {np.mean(train_loss_list[-self.args.iter_time_span*self.args.batch_size_curr:])}')
                        else:
                            # log.info(f'predictions: {preds}')
                            # log.info(f'labels: {label}')
                            log.info(
                                f'labels of a batch: {truth_labels[-self.args.batch_size_curr:]}')
                            log.info(
                                f'predictions of a batch: {predictions[-self.args.batch_size_curr:]}')
                            log.info(
                                f'Iteration {iter}: Training R2Score {r2score(torch.tensor(predictions[-self.args.iter_time_span*self.args.batch_size_curr:]), torch.tensor(truth_labels[-self.args.iter_time_span*self.args.batch_size_curr:]))}, RMSE {RMSELoss(torch.tensor(truth_labels[-self.args.iter_time_span*self.args.batch_size_curr:]), torch.tensor(predictions[-self.args.iter_time_span*self.args.batch_size_curr:]))}, Training loss {np.mean(train_loss_list[-self.args.iter_time_span*self.args.batch_size_curr:])}')
                        torch.save(modify_dict_from_dataparallel(self.model.state_dict(), self.args),
                                   f'{params_str}.pt')
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(
                    #     self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                train_loss_all_list.extend(train_loss_list)
                if self.args.output_type == 'binary' or self.args.output_type == 'categorical':
                    train_report = classification_report(
                        truth_labels, predictions, digits=4)
                    log.info('Classification report:')
                    log.info(train_report)
                else:
                    rmse_loss = RMSELoss(torch.tensor(predictions),
                                         torch.tensor(truth_labels))
                    train_report = [
                        f'Training RMSE loss: {rmse_loss}\nR2Score: {r2score(torch.tensor(predictions), torch.tensor(truth_labels))}']
                    log.info(train_report)

            self.model.eval()
            val_report, val_loss_list, correct_predictions = self.eval(
                split='val', store_csv=False)

            history['train_report'].append(train_report)
            history['train_loss'].append(np.mean(train_loss_list))
            history['val_report'].append(val_report)
            history['val_loss'].append(np.mean(val_loss_list))

            if self.args.output_type != 'real':
                cur_acc = correct_predictions/len(self.val_data)
                if cur_acc > best_acc:
                    best_history = history.copy()
                    best_acc = cur_acc
                    torch.save(modify_dict_from_dataparallel(self.model.state_dict(), self.args),
                               f'{params_str}_tmp.pt')
            else:
                if rmse_loss < best_acc:
                    best_history = history.copy()
                    best_acc = rmse_loss
                    torch.save(modify_dict_from_dataparallel(self.model.state_dict(), self.args),
                               f'{params_str}_tmp.pt')

            self.eval(split='test')
        return best_acc, best_history

    def grid_search(self):
        log.info('DfModel grid_search..')
        import torch.optim as optim
        from pytorch_transformers import WarmupLinearSchedule
        param_dict = {
            'batch_size': [int(item) for item in self.args.batch_size],
            'lr': [float(item) for item in self.args.lr],
            'dropout_rate': [float(item) for item in self.args.dropout_rate]
        }

        record_list = []
        if self.args.output_type != 'real':
            best_acc = 0
        else:
            best_acc = 100

        if self.args.output_type in ['binary', 'categorical']:
            # pass
            class_count = self.df_train[self.args.y_col].value_counts(
            ).to_dict()
            class_dict = sorted(class_count.items(), key=lambda item: item[0])
            log.info('Class labels distribution:')
            log.info(class_dict)
            if self.args.class_weight is not None:
                self.loss_f = torch.nn.CrossEntropyLoss(
                    weight=get_class_weight(
                        self.args.class_weight,
                        class_dict
                    )
                )
            else:
                self.loss_f = torch.nn.CrossEntropyLoss()
        elif self.args.output_type == 'real':
            self.loss_f = torch.nn.MSELoss()

        for dr in param_dict['dropout_rate']:
            for bs in param_dict['batch_size']:
                for lr in param_dict['lr']:
                    log.info(
                        f'----------------------------{bs}----{lr}----{dr}-----------------------------')
                    # parameters set
                    self.dataloader_params = {
                        'batch_size': bs,
                        'shuffle': True
                    }
                    self.get_dataloader()
                    self.args.dropout_rate_curr = dr

                    self.args.lr_curr = lr
                    self.args.batch_size_curr = bs
                    self.get_model()
                    log.info(self.model)

                    # if requested to continue training from a checkpoint
                    if self.args.model_load_path is not None:
                        # if loading a state_dict saved without removing extra keys in the dict,
                        # self.model.load_state_dict(modify_dict_from_dataparallel(torch.load(self.args.model_load_path)))
                        self.model.load_state_dict(
                            torch.load(self.args.model_load_path))

                    self.args = read_optimizer(
                        self.args, self.model, len(self.train_loader))
                    self.optimizer = self.args.optimizer_func
                    self.scheduler = self.args.scheduler_func

                    acc, report = self.train(
                        self.args.n_epochs, f'{self.args.model_save_dir}{self.args.now}_{self.args.output_type}_{dr}_{bs}_{lr}_{self.args.optimizer}')
                    record_list.append({'acc': acc, 'report': report})
                    if self.args.output_type == 'binary':
                        if acc.item() > best_acc:
                            best_acc = acc.item()
                            log.info(f'{bs}, {lr}, {dr}')
                    else:
                        if acc.item() < best_acc:
                            best_acc = acc.item()
                            log.info(f'{bs}, {lr}, {dr}')
        log.info('Tuning summary:')
        log.info(record_list)
