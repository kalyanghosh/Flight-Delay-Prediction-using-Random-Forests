# Flight-Delay-Prediction-using-Random-Forests
This repository hosts code and files for the project "Flight Delay Prediction"

Steps:

1. Flight data from the BTS website was downloaded for the years 2013 - 2016

2. Pandas library was used to perform basic preprocessing of data like rmoval of null values, label encoding and onehot encoding for inputs and output varaibles.

3. The data is then divided in Training and Testing data in the ratio 70:30 respectively

4. The data is fed into a Random Forest classifier

5. A 3 - fold Cross Validation is performed on the Training dataset to find the optimal set of hyperparameters for the Random Forest classifier

6. The model is retrained with the optimal set of hyperparameters

7. The performance of the model is measured by plotting the ROC curves and Confusion Matrix with the 30% Testing Data


