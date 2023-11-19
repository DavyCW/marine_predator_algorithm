# marine_predator_algorithm

1. Download the datasets from kaggle into data csv. They are too big to store on GitHub.
2. Run import_ecg_data.ipynb to get the datasets usable for the rest. First two boxes are the most important, the others are there for if you want to use different samplings. For the purposes of the example, make sure to run SMOTE Multiclass and SMOTE Multiclass to Binary.
3. Run main.ipynb to generate the optimal hyper parameters, their accuracy, and the convergence curve. You can change which data is added to the data list in order to change which datasets the models are using. F24 is Extra Trees, F25 is Random Forest, and F26 is MLP.
