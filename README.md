# Instance-weight-Balanced-Factorization-Machine (IBFM)

This repository provides a PyTorch implementation and datasets of IBFM.



## Environment

1. Python 3.7

2. PyTorch

3. Pandas

5. Sklearn



## Data description

The Frappe dataset, which has been used for context-aware recommendation, contains 93,203 app usage logs of users under different context (a.k.a feature field). Each log in this dataset includes user ID, app ID and eight context variables, such as weather, city, daytime, etc.

The MovieLens dataset, has been used for personalized tag recommendation and contains 668,953 tag applications of users on movies.



## How to Run the models

We provide the code of the IBFM model and the FM model, which to pretrain the embedding vector.



IBFM:

initialize from pretrain embedding vector by the FM model, The relate pretrain files are already in the folder pretrain.

```python
python IBFM.py -pretrain 1
```

FM:

training the FM model and save the embedding vector.

```python
python FM.py -pretrain -1
```

