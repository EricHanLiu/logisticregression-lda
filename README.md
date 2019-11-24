# Logistic Regression - LDA
Python implementations for logistic regression and linear discriminant analysis models. Trained and tested on two standard datasets.

To run, call `python main.py`. It will take some time for everything to run.

#### Final Weights
For 10,000 iterations of gradient descent training for logistic regression
(roughly same accuracy as 100,000 iterations, possibly convergence?)
> This is trained on the whole test set, no validation set left out. These weights
>can be used to quickly evaluate the accuracy on a given training set (though we only 
>have access to the original one)
```bash
[ -3536.45694032, -39222.67793879,  14792.52257357  -2229.5114057
  -4009.16627701,   3775.47642038,  -1290.86627148 -14869.66869384
 -52672.55105049,  18714.71900456,  25458.77360396]
Accuracy of: 64.2276%
```

Quadratic expansion (`x + x^2`) doesn't seem to help the logistic regression
model on the wine dataset at all (lowers accuracy), but helps the cancer
dataset (by about 5% accuracy)

Remove outliers doesn't work because the features aren't normally distributed 
(we did 3 standard deviations away from the mean), lowered accuracy

Learning rate on cancer doesn't affect accuracy because cancer converges 
really fast

Best accuracies: 
- Logistic Regression (learning rate of 0.1)
    - 0.6263322884012539 with 10000 iterations (WINE)
    - 0.6068965517241379 with 5000 iterations (WINE)
    - 0.5949843260188088 with 1000 iterations (WINE)
    - 0.5905956112852664 with 100 iterations (WINE)
    - 0.49153605015673973 with 10 iterations (WINE)
    - 0.6411764705882352 with 1000/100 iterations (CANCER)
    - 0.6529411764705882 with 10 or 1 iterations (CANCER)
- LDA
    - 0.7366771159874609 (WINE)
    - 0.9544117647058824 (CANCER)
    
Consider decreasing the learning rate over time (simulated annealing)

STATS:
- Precision, recall, sensitivity, etc. 
- Show through the p values that none of them are really normal distributions

Remove Outliers
- 0.6968253968253968 with removing outliers on cancer dataset
- 0.6411764705882352 without removing outliers
- 0.5905956112852664 without removing outliers on wine dataset
- 0.5298969072164949 with removing outliers
