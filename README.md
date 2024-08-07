# PairCFR: Enhancing Model Training on Paired Counterfactually Augmented Data through Contrastive Learning


## Contents of Repro
- [Data](#Data)
- [Models](#Models)
- [Src](#Running)
 

## Introduction

Here we provide the data including counterfactually augmented data for our methods and back_translation augmented data for comparison methods over sentiment analysis and natural language inference tasks, and codes including design of all models and combined function of contrastive and cross-entropy loss. More details can be seen in our [paper](https://arxiv.org/pdf/2406.06633).

## Data
we use human-in-loop counterfactually augmented data provided by (Kaushik et al.,2019). [counterfactually-augmented-data](https://github.com/acmi-lab/counterfactually-augmented-data.git). 
| Task | domain |calss|original to counterfactual ratio|
|----------|----------|----------|----------|
| sentiment analysis | IMDb movie reviews   | 2  | 1:1 |
| natural language inference | SNLI dataset | 3 | 1:4 |



Other test data sources：
- for sentiment analysis task
  - IMDb &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  _download from_: [https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)|
  - Amazon &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/Siki-77/amazon6_5core_polarity](https://huggingface.co/datasets/Siki-77/amazon6_5core_polarity)
  - Yelp  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/yelp_polarity](https://huggingface.co/datasets/yelp_polarity)
  - Twitter  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)
  - SST-2  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/gpt3mix/sst2/viewer/default/test](https://huggingface.co/datasets/gpt3mix/sst2/viewer/default/test)
  
- for natural language inference task:
  - SNLI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/snli](https://huggingface.co/datasets/snli)
  - MNLI-m &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/SetFit/mnli](https://huggingface.co/datasets/SetFit/mnli)
  - MNLI-mm &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_:  [https://huggingface.co/datasets/SetFit/mnli](https://huggingface.co/datasets/SetFit/mnli)
  - Negation &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/pietrolesci/stress_tests_nli](https://huggingface.co/datasets/pietrolesci/stress_tests_nli)
  - Spelling error &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_ [https://huggingface.co/datasets/pietrolesci/stress_tests_nli](https://huggingface.co/datasets/pietrolesci/stress_tests_nli)
  - Word overlap &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _download from_: [https://huggingface.co/datasets/pietrolesci/stress_tests_nli](https://huggingface.co/datasets/pietrolesci/stress_tests_nli)

[1] Kaushik D, Hovy E, Lipton Z. Learning The Difference That Makes A Difference With Counterfactually-Augmented Data[C]//International Conference on Learning Representations. 2019.

## Models

pre-trained model + classification head
list all pre-trained models used in our experiments, which can be indexed by following mode names through the HuggingFace tool：
- Bert-base-uncased
- Roberta-base
- T5-base
- Sentence-transformers/multi-qa-distilbert-cos-v1

## Running
Environment 
- python3.8
- PyTorch2.0.1

To run the code, you should install some packages and the appropriate torch version 
```python
pip install installpytorch
pip install requirement
```
Run the finetune code on IMDB CAD
```bash
cd runimdb\run
```
```python
python run_bash.py
```

Run the finetune code on SNLI CAD
```bash
cd runsnli
```
```python
python run_bash.py
```
