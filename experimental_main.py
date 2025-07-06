import src.pre_process as prep_module
from transformers import PreTrainedModel, AutoConfig
from train_baseline import train_pipeline
from src.gelectra_base import GelecTagModel
from src.gbert_base import GbertTagModel
from src.gbert_dmbdz import GBERTdmbdzTagModel
from src.xml_roberta import RobertaTagModel
from src.losses import FocalLoss, JaccardLoss
from src.sift import AdverserialLearner, hook_sift_layer
import pandas as pd
import torch

def create_model_and_config(num_labels, model_name='xlm-roberta-large'):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.hidden_dropout_prob = 0.4

    if model_name == "xlm-roberta-large":
        model = RobertaTagModel(config)
    elif model_name == 'deepset/gbert-large':
        model = GbertTagModel(config)
    elif model_name == 'dbmdz/bert-base-german-cased':
        model = GBERTdmbdzTagModel(config)
    elif model_name == 'deepset/gelectra-large':
        model = GelecTagModel(config)    

    return model, config

if __name__ == "__main__":
    train_df = pd.read_csv("./old_data/data/tagged_train.tsv", sep='\t', keep_default_na=False, na_values=None)
    prep_module.exploratory_analysis(train_df)
    preprocessed_df = prep_module.rectify_categorization(train_df.copy())
    num_labels = preprocessed_df.Tag.nunique()
    model, config = create_model_and_config(num_labels, "xlm-roberta-large")

    training_config = {
        "model_path": "deepset/gbert-large",
        "max_length": 128,
        "n_splits": 3,
        "batch_size": 32,
        "num_of_epochs": 5,
        "learning_rate": 1e-5,
        "patience": 2,
        "max_grad_norm": 1.0,
        "do_linear_scheduler": True,
        "model": model,
        "loss_fn": None,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    train_pipeline(preprocessed_df, training_config)

    