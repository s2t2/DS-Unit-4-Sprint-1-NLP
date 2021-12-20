# DS-Unit-4-Sprint-1-NLP

## Modules

  1. [Text Data](/module1-text-data/)
  2. [Vector Representations](/module2-vector-representations/)
  3. [Document Classification](/module3-document-classification/)
  4. [Topic Modeling](/module4-topic-modeling/)


## Setup

```sh
conda create -n nlp-env python=3.7 # (first time only)
conda activate nlp-env
```

```sh
pip install -r requirements.txt
```

Download the natural language model called "en_core_web_sm":

```sh
#python -m spacy download en
#python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

## Usage

### Notebooks

Run the notebook server, the view it in the browser at http://localhost:8888/tree:

```sh
jupyter notebook
```

### CLI

Module 1:

```sh
python module1-text-data/lecture.py
python module1-text-data/assignment.py
```

Module 2:

```sh
python module2-vector-representations/assignment.py
python module2-vector-representations/lecture.py
```
