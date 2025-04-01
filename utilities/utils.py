import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluate
from evaluate import load

import torch
from torch import nn
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer, TorchAoConfig)
from torchao.quantization import Int4WeightOnlyConfig
from transformers import DataCollatorWithPadding
from peft import LoraConfig, PeftModel, PeftConfig



class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def seq_class_init(check_point, num_labels, id2label, label2id, device, quantized=False):

    model = AutoModelForSequenceClassification.from_pretrained(check_point,
                                                               num_labels=num_labels,
                                                               id2label=id2label,
                                                               label2id=label2id, 
                                                              device_map=device)
    if quantized:
        quant_config = Int4WeightOnlyConfig(group_size=128)
        quantization_config = TorchAoConfig(quant_type=quant_config)
        model.config.quantization_config = quantization_config

    return model


def model_postprocessing(model, model_name):

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if 'bert' in model_name.lower():
        model.classifier = CastOutputToFloat(model.classifier)
    else:
        model.score = CastOutputToFloat(model.score)
    return model


def build_tokenizer(model_name_or_path, max_length):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation=True,
                                              padding='max_length',
                                              max_length=max_length)
    return tokenizer


def collate_func(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer)
    return data_collator


def compute_classification_metrics(eval_preds):
    metric1 = load("accuracy")
    metric2 = load("precision")
    metric3 = load("recall")
    metric4 = load("f1")
    metric5 = load("matthews_correlation")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions,
                               references=labels)["accuracy"]
    precision = metric2.compute(predictions=predictions, references=labels,
                                average="weighted")["precision"]
    recall = metric3.compute(predictions=predictions, references=labels,
                             average="weighted")["recall"]
    f1 = metric4.compute(predictions=predictions, references=labels,
                         average="weighted")["f1"]
    matthews_score = metric5.compute(
        predictions=predictions, references=labels)["matthews_correlation"]

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
               "matthews_correlation": matthews_score}


def inference_metrics(predictions, labels, model_name):
    metric1 = load("accuracy")
    metric2 = load("precision")
    metric3 = load("recall")
    metric4 = load("f1")
    metric5 = load("matthews_correlation")

    accuracy = metric1.compute(predictions=predictions,
                               references=labels)["accuracy"]
    precision = metric2.compute(predictions=predictions, references=labels,
                                average="weighted")["precision"]
    recall = metric3.compute(predictions=predictions, references=labels,
                             average="weighted")["recall"]
    f1 = metric4.compute(predictions=predictions, references=labels,
                         average="weighted")["f1"]
    matthews_score = metric5.compute(predictions=predictions, 
                                     references=labels)["matthews_correlation"]

    return pd.DataFrame([{"model_name": model_name, "accuracy": accuracy, 
                          "precision": precision, "recall": recall, "f1": f1,
                          "matthews_correlation": matthews_score}])


def inference_fn(model, dataloader, device):
    model.to(device)
    model.eval()

    preds = []
    labels = []
    
    for step, batch in enumerate(tqdm(dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            preds.extend(predictions)
            labels.extend(batch["labels"])

    return preds, labels


def macnemar_comparison(reference, predictions1, predictions2):
    mcnemar = evaluate.load("mcnemar")
    return mcnemar.compute(reference, predictions1, predictions2)


def tokenize_function(example, tokenizer):
    return tokenizer(example["text"],padding=True, truncation=True)


def lora_peft(task_type="SEQ_CLS", target_modules=None):

    config = LoraConfig(task_type=task_type,
                        r=8, lora_alpha=16, 
                        lora_dropout=0.1,
                       target_modules=target_modules)
    return config


def get_lora_model_for_seq_class(peft_path, num_labels, id2label, label2id):
    peft_model_id = peft_path
    config = PeftConfig.from_pretrained(peft_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                    num_labels=num_labels,
                                                                   id2label=id2label,
                                                                   label2id=label2id)

    base_model = model_postprocessing(base_model, config.base_model_name_or_path)
    inference_model = PeftModel.from_pretrained(base_model, peft_model_id)

    return inference_model, config


def predict(model, eval_dataloader, model_name, device):

    preds =[]
    labels = []
    model.to(device)
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        preds.extend(predictions.detach().cpu().numpy())
        labels.extend(references.detach().cpu().numpy())

    metrics = inference_metrics(preds, labels, model_name)

    return metrics, preds
