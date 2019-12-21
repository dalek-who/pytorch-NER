import torch
import torchtext
from torchtext.data import Field, LabelField, RawField
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchcrf import CRF
import re
import argparse
from tensorboardX import SummaryWriter
from functools import wraps
import random
import os
import numpy as np
from tqdm import tqdm
import logging
import json
from itertools import chain
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import pandas as pd
from collections import defaultdict
from pathlib import Path


from model import SequenceTaggingModel
import config
from utils import is_jsonable, init_logger, fig_confusion_matrix


def _wrap_handle_exception(func):
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            return result
        except KeyboardInterrupt as e:
            self._handle_fail(keyboard_interrupt=True)
            raise
        except Exception as e:
            self._handle_fail(keyboard_interrupt=False)  # 例外处理流程
            raise
    return wrapped_func


class Experiment(object):
    def __init__(self):
        self.args = self._parse_args()
        self.cm = config.ConfigModel()
        self.ct = config.ConfigTrain()
        self.cf = config.ConfigFiles(commit=self.args.commit, load_checkpoint_dir=self.args.load_checkpoint_dir)

        self._set_seed()  # 设定随机种子必须在初始化model等所有步骤之前
        self.device: torch.device = self._get_device()

        self.iter_train, \
        self.iter_valid, \
        self.iter_test, \
        self.iter_predict, \
        self.vocab_word = \
            self._get_data_iter()

        self.writer = SummaryWriter(self.cf.DIR_OUTPUT)  # 这一步自动创建了DIR_OUTPUT
        self.logger = init_logger(log_file=self.cf.out_log)
        self.best_eval_result = defaultdict(lambda: -1)

        self.model: torch.nn.Module = self._get_model(load_checkpoint_path=self.cf.load_checkpoint)
        self.model.to(self.device)
        # self.optimizer: optim.Optimizer = optim.AdamW(self.model.parameters(), lr=self.ct.learning_rate,
        #                                    eps=self.ct.adam_epsilon, weight_decay=self.ct.weight_decay)
        #
        # def lr_lambda(step):
        #     t_total = len(self.iter_train) * self.args.num_train_epochs
        #     warmup_steps = t_total * self.ct.warmup_proportion
        #     if step < warmup_steps:
        #         return float(step) / float(max(1, warmup_steps))
        #     return max(0.0, float(t_total - step) / float(max(1.0, t_total - warmup_steps)))
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.optimizer: optim.Optimizer = optim.SGD(self.model.parameters(),
                                                    lr=self.ct.learning_rate, momentum=self.ct.momentum)
        lr_lambda = lambda epoch: 1 / (1 + self.ct.lr_decay * epoch)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # 保存各种config
        with open(self.cf.out_config_files, "w") as f_cf, \
            open(self.cf.out_config_train, "w") as f_ct, \
            open(self.cf.out_config_model, "w") as f_cm, \
            open(self.cf.out_args, "w") as f_a:
            json.dump(vars(self.args), f_a, indent=4)
            json.dump(self.cf.to_jsonable_dict(), f_cf, indent=4)
            json.dump(self.ct.to_jsonable_dict(), f_ct, indent=4)
            json.dump(self.ct.to_jsonable_dict(), f_cm, indent=4)

    @_wrap_handle_exception
    def do_evaluation(self, iter_data):
        # Eval!
        self.model.eval()
        self.logger.info("***** Running evaluation/predict *****")
        tags_true, tags_pred, words_raw = [], [], []  # 真实的tag id序列，预测的tag id序列，单词原文序列
        loss_total = 0.
        for bid, batch in enumerate(tqdm(iter_data, desc="Eval")):
            with torch.no_grad():
                loss, decode = self.model.forward_for_eval(words=batch.word, tags=batch.tag, mask=batch.mask)
            loss_total += loss.item()
            tags_pred += list(chain.from_iterable(decode))
            tags_true += [self.cm.vocab_tags[tag_raw] for tag_raw in chain.from_iterable(batch.tag_raw)]
            # tags_raw_pred += [list(self.cm.vocab_tags.keys())[tag_id] for tag_id in chain.from_iterable(decode)]
            words_raw += list(chain.from_iterable(batch.word_raw))
        metrics = self.metrics(y_true=tags_true, y_pred=tags_pred)
        return {
            "loss_total": loss_total,
            "p": metrics["p"],
            "r": metrics["r"],
            "f1": metrics["f1"],
            "acc": metrics["acc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "words_raw": words_raw,
            "tags_id_pred": tags_pred
        }

    @_wrap_handle_exception
    def do_predict(self, iter_data):
        result_eval = self.do_evaluation(iter_data)
        tags_itos = list(self.cm.vocab_tags.keys())
        pair_pred = []
        for word_raw, tag_id in zip(result_eval["words_raw"], result_eval["tags_id_pred"]):
            pair_pred.append((word_raw, tags_itos[tag_id]))
        df = pd.DataFrame(data=pair_pred, columns=["word", "tag"])
        df.to_csv(self.cf.out_predict_result, sep=" ", header=False, index=False)
        self._draw_confusion_matrix(confusion_matrix=result_eval["confusion_matrix"],
                                    graph_name=self.cf.tbx_confusion_matrix_test)
        self.cf.out_success_predict.open("w").write("Predict Success!!")
        self.logger.info("*****  Predict Success !!!  *****")

    @_wrap_handle_exception
    def do_train(self, iter_train, iter_valid, load_best_after_train=True):
        self.model.train()
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)
        self.logger.info("***** Running training *****")

        global_step = 0
        best_acc = 0
        self._set_seed()
        self.model.zero_grad()
        for epoch in tqdm(range(int(self.args.num_train_epochs)), desc="Train epoch"):
            for bid, batch in enumerate(tqdm(iter_train, desc="Train batch:")):
                loss = self.model(words=batch.word, tags=batch.tag, mask=batch.mask)
                loss = loss.mean() if self.args.n_gpu > 1 else loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ct.max_grad_norm)  # 梯度剪裁，把过大的梯度限定到固定值
                self._draw_each_step(global_step=global_step, loss=loss.item(), lr=self.scheduler.get_lr()[0]) # 画图
                self.optimizer.step()
                self.model.zero_grad()
                global_step += 1
            self.scheduler.step()  # Update learning rate schedule
            eval_train = self.do_evaluation(iter_train)
            eval_valid = self.do_evaluation(iter_valid)
            eval_result = dict(**{k+"_train": v for k,v in eval_train.items()},
                               **{k+"_valid": v for k,v in eval_valid.items()})
            eval_result["epoch"] = epoch
            self._draw_each_epoch(epoch=epoch, eval_result=eval_result)# 画图
            self._create_checkpoint(epoch=epoch, eval_result=eval_result)
        if load_best_after_train:
            if self.args.num_train_epochs>0:
                self.model = self._get_model(load_checkpoint_path=self.cf.out_checkpoint)
            self.logger.info(f"**********************  Load best model of epoch [{self.best_eval_result['epoch']}]  **********************")
        self._draw_best()
        self.writer.export_scalars_to_json(self.cf.out_scalars)
        self.cf.out_success_train.open("w").write("Train Success!!")
        self.logger.info("*****  Train Success !!!  *****")

    def success(self):
        self.writer.close()
        self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / ("success-" + self.cf.DIR_OUTPUT.name))
        self.logger.info("*****  Experiment Success  *****")

    def _handle_fail(self, keyboard_interrupt=False):
        self.writer.close()
        if self.model.training:
            self._create_checkpoint(epoch=-1, eval_result=None, fail=True)
        if not keyboard_interrupt:
            self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / ("fail-" + self.cf.DIR_OUTPUT.name))

    def _draw_each_step(self,global_step, loss, lr):
        self.writer.add_scalars(self.cf.tbx_step_train_loss, {"train loss": loss}, global_step)
        self.writer.add_scalars(self.cf.tbx_step_learning_rate, {"learning rate": lr}, global_step)

    def _draw_each_epoch(self, epoch, eval_result):
        # 训练曲线
        self.writer.add_scalars(self.cf.tbx_epoch_loss, {"epoch_loss_train": eval_result["loss_total_train"],
                                                         "epoch_loss_valid": eval_result["loss_total_valid"]}, epoch)
        self.writer.add_scalars(self.cf.tbx_epoch_acc, {"epoch_acc_train": eval_result["acc_train"],
                                                        "epoch_acc_valid": eval_result["acc_valid"]}, epoch)
        self.writer.add_scalars(self.cf.tbx_epoch_f1, {"epoch_f1_train": eval_result["f1_train"],
                                                       "epoch_f1_valid": eval_result["f1_valid"]}, epoch)
        # 混淆矩阵
        self._draw_confusion_matrix(confusion_matrix=eval_result["confusion_matrix_train"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_train, global_step=epoch)
        self._draw_confusion_matrix(confusion_matrix=eval_result["confusion_matrix_valid"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_valid, global_step=epoch)

        # categories_list = list(self.cm.vocab_tags.keys())
        # fig_confusion_train = fig_confusion_matrix(confusion=eval_result["confusion_matrix_train"],
        #                                            categories_list=categories_list)
        # self.writer.add_figure(self.cf.tbx_epoch_confusion_matrix_train, fig_confusion_train, global_step=epoch)
        # fig_confusion_valid = fig_confusion_matrix(confusion=eval_result["confusion_matrix_valid"],
        #                                            categories_list=categories_list)
        # self.writer.add_figure(self.cf.tbx_epoch_confusion_matrix_valid, fig_confusion_valid, global_step=epoch)



    def _draw_best(self):
        # 混淆矩阵
        self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_train"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_train)
        self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_valid"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_valid)
        # categories_list = list(self.cm.vocab_tags.keys())
        # fig_confusion_train = fig_confusion_matrix(confusion=self.best_eval_result["confusion_matrix_train"],
        #                                            categories_list=categories_list)
        # self.writer.add_figure(self.cf.tbx_best_confusion_matrix_train, fig_confusion_train)
        # fig_confusion_valid = fig_confusion_matrix(confusion=self.best_eval_result["confusion_matrix_valid"],
        #                                          categories_list=categories_list)
        # self.writer.add_figure(self.cf.tbx_best_confusion_matrix_valid, fig_confusion_valid)

        # 最好的模型的网络图
        self._draw_model_graph(iter_data=self.iter_valid)

    def _draw_model_graph(self, iter_data):
        self.model.eval()
        batch = next(iter(iter_data))
        words = batch.word.narrow(dim=1, start=0, length=1)
        tags = batch.tag.narrow(dim=1, start=0, length=1)
        mask = batch.mask.narrow(dim=1, start=0, length=1)
        self.writer.add_graph(self.model, (words, tags, mask))

    def _draw_confusion_matrix(self, confusion_matrix: np.array, graph_name: str, global_step: int=None):
        categories_list = list(self.cm.vocab_tags.keys())
        fig_confusion = fig_confusion_matrix(confusion=confusion_matrix, categories_list=categories_list)
        if global_step is None:
            self.writer.add_figure(graph_name, fig_confusion)
        else:
            self.writer.add_figure(graph_name, fig_confusion, global_step=global_step)

    def _create_checkpoint(self, epoch, eval_result, fail=False):
        by_what = f"{self.ct.select_model_by}_valid"
        better_result: bool = eval_result is not None \
                              and eval_result[by_what] > self.best_eval_result[by_what]
        save_checkpoint: bool = fail or better_result
        checkpoint_dir = self.cf.DIR_CHECKPOINT_FAIL if fail else self.cf.DIR_CHECKPOINT
        checkpoint_file_path = self.cf.out_checkpoint_fail if fail else self.cf.out_checkpoint

        if fail:
            show_info = f"Experiment exit with Exception, checkpoint is saved at {checkpoint_file_path}"
        else:
            show_info = f'\nEpoch: {epoch} \n' + "\n".join([f' {key}: {value:.4f} ' for key, value in eval_result.items()
                                                           if isinstance(value, (int, float))])
        self.logger.info(show_info)
        if better_result:
            self.logger.info(f"\nEpoch {epoch}: {by_what} improved from {self.best_eval_result[by_what]} to {eval_result[by_what]}")
            self.logger.info("***** ***** ***** *****  save model to disk.  ***** ***** ***** *****")
            self.best_eval_result = eval_result
            with open(self.cf.out_best_eval_metrics, "w") as f:
                jsonable_best_eval_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                              for k,v in self.best_eval_result.items()}
                json.dump(jsonable_best_eval_metrics, f, indent=4)
        if save_checkpoint:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint = {
                "model_state_dict": model_to_save.state_dict(),
                "epoch": epoch,
                "best_of_which_metrics": by_what,
                "best_eval_result": dict(self.best_eval_result)}
            torch.save(checkpoint, checkpoint_file_path)


    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser()
        # 变得较多的
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_pred", action='store_true',
                            help="Whether to run eval on the test set.")

        parser.add_argument("--num_train_epochs", default=30, type=int,
                            help="Train epochs number.")
        parser.add_argument("--train_batch_size", default=32, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--eval_batch_size", default=16, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument('--train_data_num', type=int, default=None,
                            help="Use a small number to test the full code")
        parser.add_argument('--eval_data_num', type=int, default=None,
                            help="Use a small number to test the full code")

        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--load_checkpoint_dir', type=str, default="",
                            help="Whether to use checkpoints to load model. If not given checkpoints, implement a new model")
        parser.add_argument('--commit', type=str, default='',
                            help="Current experiment's commit")
        args = parser.parse_args()
        return args

    def _get_device(self):
        device = torch.device("cuda") if torch.cuda.is_available() and not self.args.no_cuda else torch.device("cpu")
        self.args.n_gpu = torch.cuda.device_count()
        return device

    def _get_data_iter(self):
        WORD = Field(sequential=True, use_vocab=True, lower=False, pad_token="<pad>", unk_token="<unk>")
        WORD_RAW = RawField()  # 纯tag标记，变成batch化时不会被转成tensor
        TAG = Field(sequential=True, use_vocab=False, lower=False, pad_token=-1, unk_token=None,
                    preprocessing=lambda tag_list: [self.cm.vocab_tags[tag] for tag in tag_list])
        TAG_RAW = RawField()  # 纯tag标记，变成batch化时不会被转成tensor
        SEQ_LEN = Field(sequential=True, use_vocab=False,
                    preprocessing=lambda word_list: [len(word_list)])
        MASK = Field(sequential=True, use_vocab=False, pad_token=0,
                     preprocessing=lambda word_list: [1] * len(word_list))

        fields = [(("word", "seq_len", "mask", "word_raw"), (WORD, SEQ_LEN, MASK, WORD_RAW)),
                  (None, None),
                  (None, None),
                  (("tag", "tag_raw"), (TAG, TAG_RAW))]

        train, valid, test = SequenceTaggingDataset.splits(
            path=self.cf.DIR_DATA, fields=fields, separator=" ",
            train=self.cf.data_train.name, validation=self.cf.data_valid.name, test=self.cf.data_test.name)

        if self.args.do_train:
            if self.args.train_data_num is not None:
                train.examples = train.examples[:self.args.train_data_num]
            if self.args.eval_data_num is not None:
                valid.examples = valid.examples[:self.args.eval_data_num]

        if self.args.do_pred and self.args.eval_data_num is not None:
            test.examples = test.examples[:self.args.eval_data_num]

        WORD.build_vocab(train.word, vectors=[GloVe(name='6B', dim=self.cm.embedding_dim, cache=self.cf.DIR_W2V_CACHE)])

        iter_train, iter_valid = torchtext.data.BucketIterator.splits(
            (train, valid), batch_sizes=(self.args.train_batch_size, self.args.eval_batch_size), device=self.device,
            sort_key=lambda x: len(x.word), sort_within_batch=True)

        iter_test = torchtext.data.Iterator(
            test, batch_size=self.args.eval_batch_size, device=self.device,
            sort=False, shuffle=False, sort_within_batch=False, train=False)

        vocab_word = WORD.vocab
        iter_predict = iter_test

        return iter_train, iter_valid, iter_test, iter_predict, vocab_word

    def _get_model(self, load_checkpoint_path: str or Path = None):
        model: nn.Module = SequenceTaggingModel(
            vocab=self.vocab_word, embedding_dim=self.cm.embedding_dim, lstm_hidden_size=self.cm.lstm_hidden_size,
            lstm_layers=self.cm.lstm_layers, tags_num=self.cm.tags_num, dropout=self.cm.dropout,
            dropout_embed=self.cm.dropout_embed)
        if load_checkpoint_path is not None:
            checkpoint = torch.load(load_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _set_seed(self):
        """
        设置所有的随机种子
        :return:
        """
        seed = self.ct.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def metrics(self, y_true, y_pred):
        p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        confusion_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
        return {
            "p": p,
            "r": r,
            "f1": f1,
            "acc": acc,
            "confusion_matrix": confusion_mtx,
        }


if __name__ == '__main__':
    experiment = Experiment()
    if experiment.args.do_train:
        experiment.do_train(iter_train=experiment.iter_train,
                            iter_valid=experiment.iter_valid)
    if experiment.args.do_pred:
        experiment.do_predict(iter_data=experiment.iter_test)
    experiment.success()
