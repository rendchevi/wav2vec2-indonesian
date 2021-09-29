#!/usr/bin/env python3
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torchaudio
from packaging import version
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process


if is_apex_available():
    from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    project_name: str = field(
        metadata={"help": "Name of project for Wandb logging"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    freeze_all_but_cls: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze all layers but the linear layer."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_data_files: str = field(
        metadata={"help": "Path to metadata of train files"}
    )
    eval_data_files: str = field(
        metadata={"help": "Path to metadata of eval files"}
    )
    vocab_dict_path: str = field(
        metadata={"help": "Path to vocab dictionary"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train+validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    chars_to_ignore: List[str] = list_field(
        default=[",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�"],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )


class PrepDataset:
    
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        master_path: str,
        chars_to_ignore_regex: str,
    ):

        self.processor = processor
        self.master_path = master_path
        self.chars_to_ignore_regex = chars_to_ignore_regex
    
    def __call__(
        self,
        batch,
    ):
        # Strip unnecessary chars in sentences
        batch["sentence"] = re.sub(self.chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        
        # Tokenize sentences
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["sentence"]).input_ids
        
        # Append complete wav path
        batch["input_values"] = os.path.join(self.master_path, "wavs", batch["path"])

        return batch


@dataclass
class DataCollatorCTCWithPadding:

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):

        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": self.processor(torchaudio.load(feature["input_values"])[0][0].numpy(), sampling_rate = 16_000).input_values[0].tolist() } for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

def get_processor(
        vocab_dict_path,
    ):
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_dict_path,
        unk_token = "[UNK]",
        pad_token = "[PAD]",
        word_delimiter_token = "|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size = 1,
        sampling_rate = 16_000,
        padding_value = 0.0,
        do_normalize = True,
        return_attention_mask = True
    )

    processor = Wav2Vec2Processor(feature_extractor = feature_extractor, tokenizer = tokenizer)
    
    return tokenizer, feature_extractor, processor

def get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                         num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        constant_steps = int(num_training_steps * 0.4)
        warmup_steps = int(num_training_steps * 0.1)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps+constant_steps:
            return 1
        else: return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps+constant_steps)))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_flat_scheduler(
    name=None,
    optimizer=None,
    num_warmup_steps=None,
    num_training_steps=None,
):
    return get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)


class FlatTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def create_flat_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_flat_scheduler(optimizer = self.optimizer,
                                              num_training_steps=num_training_steps)
    
    def create_optimizer_and_scheduler(self, num_training_steps):
        self.create_optimizer()
        self.create_flat_scheduler(num_training_steps)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_ENTITY"] = "kata-research"
    os.environ["WANDB_PROJECT"] = data_args.project_name
    os.environ["WANDB_LOG_MODEL"] = "true"

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    """
    ======================
    1] LOAD RAW DATASET
    ======================
    """

    # Load raw train dataset (contain wav path and sentence)
    train_dataset = datasets.load_dataset(
        "csv",
        data_files = data_args.train_data_files,
        column_names = ["path", "sentence"],
        delimiter = "|",
        cache_dir = model_args.cache_dir,
    )["train"]
    
    # Load raw val dataset
    eval_dataset = datasets.load_dataset(
        "csv",
        data_files = data_args.eval_data_files,
        column_names = ["path", "sentence"],
        delimiter = "|",
        cache_dir = model_args.cache_dir,
    )["train"]

    print(
        f"[Train Data]: {train_dataset.num_rows} [Dev Data]: {eval_dataset.num_rows}"
    )

    """
    ======================
    2] PRE-PROCESS DATASET
    ======================
    """

    tokenizer, feature_extractor, processor = get_processor(data_args.vocab_dict_path)

    # Apply preprocessing function
    train_dataset = train_dataset.map(PrepDataset(
        processor = processor,
        master_path = data_args.train_data_files.split("/")[:-1],
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]',
    ))
    train_dataset = train_dataset.shuffle()

    eval_dataset = eval_dataset.map(PrepDataset(
        processor = processor,
        master_path = data_args.eval_data_files.split("/")[:-1],
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]',
    ))


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=model_args.gradient_checkpointing,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=True
    )

    # Metric
    wer_metric = datasets.load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    if model_args.freeze_feature_extractor == True:
        model.freeze_feature_extractor()

    if model_args.freeze_all_but_cls == True:
        for p in model.parameters():
            p.requires_grad = False
        model.lm_head.parameters().requires_Grad = True

    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    fe_param = sum(p.numel() for p in model.wav2vec2.feature_extractor.parameters())
    fp_param = sum(p.numel() for p in model.wav2vec2.feature_projection.parameters())
    enc_param = sum(p.numel() for p in model.wav2vec2.encoder.parameters())
    cls_param = sum(p.numel() for p in model.lm_head.parameters())

    print(f"Feature Encoder Params  : {fe_param:,}")
    print(f"Feature Projector Params: {fp_param:,}")
    print(f"Encoder Params          : {enc_param:,}")
    print(f"Classifier Head Params  : {cls_param:,}")
    print("===============================================")
    print(f"Total Params            : {total_param:,}")
    print(f"Trainable Param         : {trainable_param}")
    print("===============================================")

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize our Trainer    
    trainer = FlatTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )
    
    # save the feature_extractor and the tokenizer
    if is_main_process(training_args.local_rank):
        processor.save_pretrained(training_args.output_dir)

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    main()
