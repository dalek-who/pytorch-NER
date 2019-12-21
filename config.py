"""
实验过程的config
"""
#%%
from dataclasses import dataclass
from pathlib import Path
import socket
from datetime import datetime
import os\

from utils import is_jsonable


class Config(object):
    def to_jsonable_dict(self):
        return  {k: v if is_jsonable(v) else str(v) for k, v in vars(self).items()}

class ConfigTrain(Config):
    learning_rate: float = 0 #0.015
    lr_decay: float = 0.05
    max_grad_norm: float = 5.0
    momentum: float = 0.9
    # weight_decay: float = 0.1
    # adam_epsilon: float = 1e-8
    # warmup_proportion: float = 0.1
    seed: int = 42
    select_model_by: str = "f1"
    optimizer: str = "SGD"


class ConfigModel(Config):
    embedding_dim: int = 300
    lstm_hidden_size: int = 200
    lstm_layers: int = 2
    dropout: float = 0.5
    dropout_embed: float = 0.5
    tags_num: int = 8
    vocab_tags: dict = {'O': 0, 'I-PER': 1, 'I-ORG': 2, 'I-LOC': 3, 'I-MISC': 4, 'B-MISC': 5, 'B-ORG': 6, 'B-LOC': 7}


class ConfigFiles(Config):
    def __init__(self, commit: str="", load_checkpoint_dir: str=""):
        self.DIR_BASE = Path(".")
        self.DIR_DATA = self.DIR_BASE / "data"
        self.DIR_LOAD_CHECKPOINT: Path = self.DIR_BASE / load_checkpoint_dir if load_checkpoint_dir.strip() else None
        self.DIR_W2V_CACHE: Path = self.DIR_BASE / 'vector_cache'
        self.DIR_OUTPUT = self.DIR_BASE / f"output/{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}__{commit}"
        self.DIR_CHECKPOINT = self.DIR_OUTPUT / "checkpoint"
        self.DIR_CHECKPOINT_FAIL = self.DIR_OUTPUT / "X_checkpoint_fail"
        self.DIR_TENSORBOARD: Path = self.DIR_OUTPUT / "tbX"

        # inputs:
        #   model checkpoint
        self.load_checkpoint: Path = self.DIR_LOAD_CHECKPOINT / "checkpoint.pth" if self.DIR_LOAD_CHECKPOINT else None
        #   data
        self.data_train: Path = self.DIR_DATA / 'eng.train'
        self.data_valid: Path = self.DIR_DATA / 'eng.testa'
        self.data_test: Path = self.DIR_DATA / 'eng.testb'
        self.data_pred: Path = self.DIR_DATA / 'eng.testb'

        # outputs:
        self.out_checkpoint: Path = self.DIR_CHECKPOINT / "checkpoint.pth"
        self.out_checkpoint_fail: Path = self.DIR_CHECKPOINT_FAIL / "checkpoint.pth"
        self.out_predict_result: Path = self.DIR_OUTPUT / "predict_tag.txt"
        self.out_log: Path = self.DIR_OUTPUT / "log.txt"
        self.out_args: Path = self.DIR_OUTPUT / "args.json"
        self.out_config_files: Path = self.DIR_OUTPUT / "config_files.json"
        self.out_config_model: Path = self.DIR_OUTPUT / "config_model.json"
        self.out_config_train: Path = self.DIR_OUTPUT / "config_train.json"
        self.out_best_eval_metrics: Path = self.DIR_OUTPUT / "best_eval_metrics.json"
        self.out_scalars: Path = self.DIR_OUTPUT / "all_scalars.json"
        self.out_success_train: Path = self.DIR_OUTPUT / "zzz_SUCCESS_train.txt"
        self.out_success_predict: Path = self.DIR_OUTPUT / "zzz_SUCCESS_predict.txt"

        # tensorboardX records:
        self.tbx_step_train_loss: str = "tbX/step_train_loss"
        self.tbx_step_learning_rate: str = "tbX/step_learning_rate"
        self.tbx_epoch_loss: str = "tbX/epoch_loss"
        self.tbx_epoch_acc: str = "tbX/epoch_acc"
        self.tbx_epoch_f1: str = "tbX/epoch_f1"
        self.tbx_epoch_confusion_matrix_train: str = "tbX/confusion_matrix/epoch_train"
        self.tbx_epoch_confusion_matrix_valid: str = "tbX/confusion_matrix/epoch_valid"
        self.tbx_best_confusion_matrix_train: str = "tbX/confusion_matrix/best_train"
        self.tbx_best_confusion_matrix_valid: str = "tbX/confusion_matrix/best_valid"
        self.tbx_confusion_matrix_test: str = "tbX/confusion_matrix/test"



