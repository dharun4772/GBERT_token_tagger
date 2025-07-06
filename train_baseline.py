from src.gelectra_base import GelecTagModel
from src.gbert_base import GbertTagModel
from src.gbert_dmbdz import GBERTdmbdzTagModel
from src.xml_roberta import RobertaTagModel
from src.losses import FocalLoss, JaccardLoss
from src.sift import AdverserialLearner, hook_sift_layer
import random
import numpy as np
import torch
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
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
        return tokenized

def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

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

        category_scores = []

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

