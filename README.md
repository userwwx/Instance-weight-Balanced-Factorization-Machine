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

We follow the detailed data preprocessing operations of [NFM](https://github.com/hexiangnan/neural_factorization_machine), which randomly split the Frappe and MovieLens datasets into the training (80%), validation (10%), test (10%) subsets, convert each row of data to a feature vector using the one-hot encoding, and sample two negative instances to pair with one positive instance to ensure the generalization of the model for both datasets.

## Example to run the codes.

We provide the codes (see folder code/) of the IBFM model and the FM model, which to pretrain the feature weight. 



IBFM:

initialize from pretrain feature weight by the FM model, The relate pretrain files are already in the folder pretrain/.

```python
python IBFM.py -pretrain 1 -dataset frappe -metric RMSE
```

The instruction of commands has been clearly stated in the codes.

FM:

training the FM model and save the feature weight.

```python
python FM.py -pretrain -1
```

