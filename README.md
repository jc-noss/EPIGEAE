# EPIGEAE



Source code for Findings of ACL 2025 paper: [Event Pattern-Instance Graph: A Multi-Round Role Representation
Learning Strategy for Document-Level Event Argument Extraction].

Our code is based on PAIE [here](https://github.com/mayubo2333/PAIE) and thanks for their implement.


<div align=center>
<img width="800" height="350" src="./figure/model.png"/>
</div>

## 🚀 How to use our code?

# Project Title

This project provides the implementation of **EPIGEAE**, with support for dataset processing, training, and evaluation.

## 1. Setup Environment

Before starting, ensure you have the required dependencies and environment set up.

### 1.1 Create Conda Environment

First, create a new Conda environment with Python 3.7:

### 1.2 Install Dependencies

After activating the environment, install all necessary packages:

```bash
pip install -r requirements.txt
```

### 1.3 Install SpaCy Language Model

To enable language processing with SpaCy, download the `en_core_web_sm` model using the following command:

```bash
python -m spacy download en_core_web_sm
```

---

## 2. Data Preprocessing

You can refer to the **PAIE** project [here](https://github.com/mayubo2333/PAIE) to obtain the datasets. 

## 3. Training and Inference

The following scripts can be used to train and evaluate models on different datasets:

### 3.1 Training

You can train models by running the corresponding scripts:

```bash
bash ./scripts/train_wikievent.sh
bash ./scripts/train_rams.sh
bash ./scripts/train_oeecfc.sh
```

Each script is tailored to a specific dataset and configuration. You may modify the settings inside these scripts to suit your needs.



## 🌝 Citation