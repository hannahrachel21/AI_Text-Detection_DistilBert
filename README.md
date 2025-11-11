# AI-Generated Text Detection (Research Domain)
This project is a **DistilBERT-based AI text detection system** designed to distinguish between **AI-generated** and **human-written** research texts.

It improves upon the earlier **AuthenText** model, which used a simple feedforward neural network but couldn’t capture contextual meaning.  
This version leverages **transformer-based context learning** and includes **LIME explainability** plus a **FastAPI web interface**.

## Features
- Focused on **research-domain** texts only  
- Transformer-based classification using `DistilBERT`
- Integrated **LIME** for model interpretability
- **FastAPI UI** for real-time prediction + explanation
- **Mixed precision** for optimized GPU training

## Training Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.59% |
|Testing Accuracy | 97.4% |
| Epochs | 4 |
| Model | DistilBERT-base-uncased |
| Dataset | 80,000 |

## Dataset Information
The datasets used are research-labelled-abstracts.csv and wiki-labelled.csv, which is then preprocessed and combined to form ai_human.csv. The later dataset is used for training and testing purposes.
