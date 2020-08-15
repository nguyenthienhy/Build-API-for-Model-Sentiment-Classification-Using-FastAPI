import pandas as pd
from app.service.models import *
from tqdm import tqdm
from torch import nn
import json
import numpy as np
import pickle
from torch.nn.functional import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import *
import torch
import torch.utils.data
import argparse
from transformers.modeling_utils import *
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from app.service.utils import *
from app.service import constant
from app.service import paths

# Load BPE encoder 
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default=paths.bpe_codes_path,
    required=False,
    type=str,  
    help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)

vn_tokenizer = VnCoreNLP(paths.vncore_jar_path,
                         annotators="wseg", max_heap_size='-Xmx500m')

seed_everything(69)

# Load model
config = RobertaConfig.from_pretrained(
    paths.config_path,
    output_hidden_states = True,
    num_labels = 3
)

model_bert = RobertaForAIViVN.from_pretrained(paths.pretrained_path, config=config)
model_bert.cuda()

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file(paths.dict_path)

# Load training data
with open(paths.x_train_file_path, "rb") as f:
    X_train = pickle.load(f)
X_train = convert_lines(X_train, vocab, bpe, constant.max_sequence_length)
with open(paths.y_train_file_path, "rb") as f:
    y_train = pickle.load(f)

# Creating optimizer and lr schedulers
param_optimizer = list(model_bert.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(constant.epochs * X_train.shape[0] / constant.batch_size / constant.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr = constant.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler
loss_t = torch.nn.CrossEntropyLoss() # Define loss function

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y_train))

for fold, (train_idx, val_idx) in enumerate(splits):
    print("Training for fold {}".format(fold))
    best_score = 0
    if fold != constant.fold:
        continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long), torch.tensor(y_train[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(y_train[val_idx],dtype=torch.long))
    tq = tqdm(range(constant.epochs + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False
    frozen = True
    for epoch in tq:

        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        val_preds = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=constant.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=constant.batch_size, shuffle=False)
        avg_loss = 0.
        avg_accuracy = 0.

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            model_bert.train()
            y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            loss =  loss_t(y_pred.view(-1, 3).cuda(),y_batch.long().cuda())
            loss = loss.mean()
            loss.backward()
            if i % constant.accumulation_steps == 0 or i == len(pbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            lossf = loss.item()
            pbar.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)

        model_bert.eval()
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        all_preds = []
        for i,(x_batch, y_batch) in pbar:
            logits = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            predictions = torch.argmax(softmax(logits, 1), 1)
            all_preds.extend(predictions.cpu())
        all_preds = [it.item() for it in all_preds]
        score = f1_score(y_train[val_idx], np.array(all_preds), average='macro')
        print("F1: {}".format(score))
        if score >= best_score:
            torch.save(model_bert.state_dict(),paths.model_path)
            best_score = score
