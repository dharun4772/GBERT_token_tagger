import random
import numpy as np
import torch
import os
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from dotenv import load_dotenv
load_dotenv()

import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def tokenize(example, tokenizer, label2id, max_length, method='first_subword'):
    if method == 'first_subword':
        tokenized = tokenizer(
            example['tokens'],
            truncation=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_tensors='pt',
            max_length=max_length
        )
        word_ids = tokenized.word_ids()
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(label2id(example['labels'][word_id]))
            else:
                labels.append(-100)
            prev_word_id = word_id
        tokenized['labels'] = torch.tensor(labels)
        tokenized['categories'] = torch.tensor(example['categories'][0])
        return tokenized

def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

class fbetaScore:
    def __init__(self,
                 predicted_aspects: List[Tuple[int, int]],
                 labeled_aspects: List[Tuple[int, int]], 
                 beta: float = 0.2) -> Tuple[Dict[int, int], float]:
        self.pred_dict = defaultdict(set)
        self.label_dict = defaultdict(set)
        self.predicted_aspects = predicted_aspects
        self.labeled_aspects = labeled_aspects
        self.beta = beta
    
    def compute_scores(self,):
        for idx, (aspect, category) in enumerate(self.predicted_aspects):
            self.pred_dict[(aspect, category)].add(idx)
        for idx, (aspect, category) in enumerate(self.labeled_aspects):
            self.label_dict[(aspect, category)].add(idx)

        aspect_names = set(aspect for aspect, _ in self.predicted_aspects + self.labeled_aspects)
        categories = set(cat for _, cat in self.predicted_aspects + self.labeled_aspects)

        category_scores = {}

        for category in categories:
            total_labeled_aspects_in_cat = sum(
                len(self.label_dict[(aspect, category)]) for aspect in aspect_names
            )

            if total_labeled_aspects_in_cat == 0:
                category_scores[category] = 0.0
                continue

            weighted_f_scores = []
            for aspect in aspect_names:
                key = (aspect, category)
                pred_set = self.pred_dict.get(key, set())
                label_set = self.label_dict.get(key, set())

                intersection = pred_set & label_set
                precision = len(intersection) / len(pred_set) if pred_set else 0.0
                recall = len(intersection) / len(label_set) if label_set else 0.0
                f_beta = fbeta_score(precision, recall, self.beta)

                weight = len(label_set) / total_labeled_aspects_in_cat
                weighted_f_scores.append(weight * f_beta)

            category_scores[category] = sum(weighted_f_scores)

        final_avg_fbeta = np.mean(list(category_scores.values())) if category_scores else 0.0
        return category_scores, final_avg_fbeta


class ERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tokenize(self.data[idx], self.tokenizer, self.label2id, self.max_length)


def train_pipeline(preprocessed_df, config):
    sentences = defaultdict(list)
    tags = defaultdict(list)
    categories = defaultdict(list)
    for _, row in preprocessed_df.iterrows():
        rec_id = row["Record Number"]
        sentences[rec_id].append(row["Token"])
        tags[rec_id].append(row["Tag"])
        categories[rec_id].append(row["Category"])
    data = [{"tokens": tokens, "labels": labels, 'categories':cats} for tokens, labels, cats in zip(sentences.values(), tags.values(), categories.values())]
    
    label_list = sorted(preprocessed_df.Tag.unique())
    label2id =  {label:idx for idx, label in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(config.get("model_path"))
    dataset = ERDataset(data, tokenizer, label2id, config.get("max_length"))
    kfold = KFold(n_splits=config.get("n_splits"), shuffle=True, random_state=42)
    collator = DataCollatorForTokenClassification(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        wandb.init(
            project="token-classification-german-ebay",
            name=f"{config['model_path']}_fold{fold}",
            config=config
        )
        print(f"Fold {fold}:")
        print(f"Train indices: {train_idx[:5]}")
        print(f"Validation indices: {val_idx[:5]}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        print(f"Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")
        trainloader = DataLoader(train_subset, batch_size=config.get("batch_size", 32), shuffle=True, collate_fn=collator)
        valloader = DataLoader(val_subset, batch_size=config.get("batch_size", 32), shuffle=False, collate_fn=collator)
        num_epochs = config.get("num_of_epochs", 6)
        model = config.get("model")
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.get("learning_rate",1e-5))
        loss_fn = config.get("loss_fn", nn.CrossEntropyLoss(ignore_index=-100))
        scheduler = None

        if config.get("do_linear_scheduler", None):
            total_steps = len(trainloader)*num_epochs
            warmup_steps = int(0.1*total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_training_steps=len(trainloader),
                num_warmup_steps=total_steps,
            )

        best_val_fbeta = 0
        patience = config.get("patience", 2)
        patience_counter = 0
        best_model_state = None

        max_grad_norm = config.get("max_grad_norm", 1.0)

        example_ct = 0
        for epoch in range(num_epochs):

            model.train()
            total_loss = 0
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            all_preds = []
            all_labels = []
            all_predicted_aspects = []
            all_labeled_aspects = []
            for batch in tqdm(trainloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_masks = batch['attention_masks'].squeeze(1).to(device)
                labels = batch['labels'].to(device)
                categories = batch['categories'].to(device)

                outputs = model(input_ids, attention_masks, labels = labels)
                # loss = loss_fn(outputs.view(-1, len(label_list)), labels.view(-1))
                loss = outputs.loss
                logits = outputs.logits 

                outputs_map = []
                labels_map = []
                for idx, output in enumerate(logits):
                    mod_output = torch.argmax(output, dim=1)
                    for jdx, out in enumerate(mod_output):
                        if labels[idx][jdx]!=-100:
                            outputs_map.append((int(out.cpu()), int(categories[idx].cpu())))
                            labels_map.append((int(labels[idx][jdx].cpu()), int(categories[idx].cpu())))

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                example_ct += len(labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                preds_flat = preds.view(-1)
                labels_flat = labels.view(-1)
                mask = labels_flat != -100
                filtered_preds = preds_flat[mask]
                filtered_labels = labels_flat[mask]
                all_preds.extend(filtered_preds.cpu().numpy())
                all_labels.extend(filtered_labels.cpu().numpy())
                all_predicted_aspects.extend(outputs_map)
                all_labeled_aspects.extend(labels_map)
            accuracy = accuracy_score(all_labels, all_preds)
            category_scores, fbeta_scores = fbetaScore(all_predicted_aspects, all_labeled_aspects, beta=0.2).compute_scores()
            avg_loss = total_loss / len(trainloader)
            print(f"Token-Level Training Accuracy after Epoch {epoch+1}: {accuracy:.4f}")
            print(f"Average Loss after Epoch {epoch+1}: {avg_loss:.4f}")
            print(f"Fbeta scores per category after Epoch {epoch+1}: {fbeta_scores}")
            current_lr = scheduler.get_last_lr()[0] if scheduler else config.get("learning_rate", 1e-5)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": accuracy,
                "train_fbeta_avg": fbeta_scores,
                "learning_rate": current_lr,
                **{f"train_fbeta_{cat}": score for cat, score in fbeta_scores.items()}
            })


            model.eval()
            val_loss=0
            all_preds = []
            all_labels = []
            all_predicted_aspects = []
            all_labeled_aspects = []
            with torch.no_grad():
                for i, batch in enumerate(tqdm(valloader)):
                    input_ids = batch['input_ids'].squeeze(1).to(device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(device)
                    labels = batch['labels'].to(device)
                    categories = batch['categories'].to(device)
                    outputs = model(input_ids, attention_mask, labels = labels)
                    # loss = loss_fn(outputs.view(-1, len(label_list)), labels.view(-1))
                    loss = outputs.loss
                    logits = outputs.logits
                    val_loss += loss.item()

                    outputs_map = []
                    labels_map = []
                    for idx, output in enumerate(logits):
                        mod_output = torch.argmax(output, dim=1)
                        for jdx, out in enumerate(mod_output):
                            if labels[idx][jdx]!=-100:
                                outputs_map.append((int(out.cpu()), int(categories[idx].cpu())))
                                labels_map.append((int(labels[idx][jdx].cpu()), int(categories[idx].cpu())))

                    preds = torch.argmax(logits, dim=-1)
                    preds_flat = preds.view(-1)
                    labels_flat = labels.view(-1)
                    mask = labels_flat != -100
                    filtered_preds = preds_flat[mask]
                    filtered_labels = labels_flat[mask]
                    all_preds.extend(filtered_preds.cpu().numpy())
                    all_labels.extend(filtered_labels.cpu().numpy())
                    all_predicted_aspects.extend(outputs_map)
                    all_labeled_aspects.extend(labels_map)

            avg_val_loss = val_loss / len(valloader)
            accuracy = accuracy_score(all_labels, all_preds)
            category_scores, val_fbeta_scores = fbetaScore(all_predicted_aspects, all_labeled_aspects, beta=0.2).compute_scores()
            print(f"Token-Level Validation Accuracy after Epoch {epoch+1}: {accuracy:.4f}")
            print(f"Average Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
            print(f"Fbeta scores per category after Epoch {epoch+1}: {fbeta_scores}")
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
                "val_fbeta_avg": val_fbeta_scores,
                **{f"val_fbeta_{cat}": score for cat, score in fbeta_scores.items()}
            })

            if val_fbeta_scores>best_val_fbeta:
                best_val_fbeta = val_fbeta_scores
                patience_counter=0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break 
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model_path = f"./ebay_training_experiments/model_fold{fold}.pt"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)
        wandb.finish()