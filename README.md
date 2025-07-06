# German Token Tagger for eBay Listings

This project implements a token classification system for German eBay product listings, utilizing various transformer-based models to extract structured information from unstructured listing titles. The system uses BIO (Beginning, Inside, Outside) tagging scheme to identify and classify different components of product listings such as brand names, model numbers, specifications, etc.

## Project Structure

```
.
├── src/                       # Source code directory
│   ├── gbert_base.py         # GBERT large model implementation
│   ├── gbert_dmbdz.py        # BERT German base model (DBMDZ) implementation
│   ├── gelectra_base.py      # German ELECTRA model implementation
│   ├── xml_roberta.py        # XLM-RoBERTa model implementation
│   ├── losses.py             # Custom loss functions (Focal, Jaccard)
│   └── sift.py               # SIFT adversarial learning implementation
├── configs/                   # Configuration files
├── data/                     # Data directory
├── pre_process.py            # Data preprocessing utilities
├── extraction.py             # Data extraction scripts
├── train_baseline.py         # Training pipeline
└── README.md                 # Project documentation
```

## Features

- Multiple transformer model architectures supported:
  - GBERT (German BERT Large)
  - DBMDZ BERT German
  - German ELECTRA
  - XLM-RoBERTa
- Advanced training techniques:
  - Multi-sample dropout
  - LSTM layer option
  - Mean pooling option
  - Custom loss functions (Focal Loss, Jaccard Loss)
  - SIFT adversarial learning
- BIO tagging scheme for token classification
- Comprehensive preprocessing pipeline
- F-beta score evaluation metrics

## Supported Tags

The model can identify various product attributes including:
- Kompatible_Fahrzeug_Marke (Compatible Vehicle Brand)
- Kompatibles_Fahrzeug_Modell (Compatible Vehicle Model)
- Herstellernummer (Manufacturer Number)
- Produktart (Product Type)
- Hersteller (Manufacturer)
- And many more product-specific attributes

## Getting Started

1. Setup Environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

2. Data Preparation:
```bash
python extraction.py  # Extract data from compressed files
python pre_process.py  # Preprocess and analyze the data
```

3. Training:
```bash
python train_baseline.py
```

## Model Architecture

Each model implementation follows a similar architecture with customizable components:
- Base Transformer (BERT/ELECTRA/RoBERTa)
- Optional LSTM layer
- Multi-sample dropout (5 different rates)
- Linear classification layer

## Advanced Features

### Multi-sample Dropout
The implementation uses 5 different dropout rates (0.1-0.5) and averages their predictions for better generalization.

### SIFT Adversarial Learning
Implements Smart and Interpretable Feature Transformation (SIFT) for adversarial training, improving model robustness.

### Custom Loss Functions
- Focal Loss: For handling class imbalance
- Jaccard Loss: For optimizing overlap between predicted and true tags

## Performance Metrics

The model performance is evaluated using:
- F-beta score (configurable beta)
- Category-wise weighted metrics
- Custom evaluation for aspect-based token classification

## License


## Contributors

dharun4772 (github/dharun4772)