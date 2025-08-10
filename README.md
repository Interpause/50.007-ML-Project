# Lol we will have to rewrite this README later anyways for the TA

Useful command to launch a jupyter server that does serverside notebook rendering and hence never loses state even after being closed:

```sh
uv run jupyverse --set frontend.collaborative=true --set auth.mode=noauth --set kernels.require_yjs=true --set jupyterlab.server_side_execution=true --port 8349
```

# 50.007 ML Project

- Kaggle link: <https://www.kaggle.com/competitions/50-007-machine-learning-summer-2024>
- Rubrics: <https://www.kaggle.com/competitions/50-007-machine-learning-summer-2024/overview/grading-metric>

`task1.ipynb` and `task2.ipynb` is expected to be relatively simple to do, so one person can take each. `task3/` is a folder because part of the deliverables is every method we tried. So everyone can try different methods as separate notebooks inside the folder.

**Below** are details copied from the Kaggle:

## Project Summary

(lol) The detection of online hate speech can be formulated as a text classification task: "Given a social media post, classify if the post is hateful or non-hateful". In this project, you are to apply machine learning approaches to perform hate speech classification. Specifically, you will need to perform the following tasks.

## Task 0: Document your Journey & Thoughts (5 marks)

All good projects must come to an end. You will need to document your machine learning journey in your final report. Specifically, please include the following in your report:

- An introduction of your best performing model (how it works)
- How did you "tune" the model. Discuss the parameters that you have used and the different parameters that you have tried before arriving at the best results.
- Did you self-learned anything that is beyond the course? If yes, what are they, and do you think if it should be taught in future Machine Learning courses.

### Deliverables

- A final report (in PDF) answering the above questions.

## Task 1: Implement Logistics Regression (Basic)

Implement Logistic Regression **from scratch**. **DO NOT USE** sklearn logistic regression or any other pre-defined logistic regression package.

### Deliverables

- Code implementation of the Logistic Regression modele.
- Predictions made by Logistic Regression model on Test set named as `LogRed_Prediction.csv`. (I copied the typo from Kaggle)

### Tips

- Check out the Logistic Regression implementation from: <https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2>
- Your implementation should have the following functions:
  - `sigmoid(z)`: A function that takes in a Real Number input and returns an output value between 0 and 1.
  - `loss(y, y_hat)`: A loss function that allows us to minimize and determine the optimal parameters. The function takes in the actual labels y and the predicted labels y_hat, and returns the overall training loss. Note that you should be using the Log Loss function taught in class.
  - `gradients(X, y, y_hat)`: The Gradient Descent Algorithm to find the optimal values of our parameters. The function takes in the training feature X, actual labels y and the predicted labels y_hat, and returns the partial derivative of the Loss function with respect to weights (w) and bias (db).
  - `train(X, y, bs, epochs, lr)`: The training function for your model.
  - `predict(X)`: The prediction function where you can apply your validation and test sets.

## Task 2: Apply Dimension Reduction Techniques (Basic)

Dimension reduction is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data. The train dataset contains 5000 TD-IDF features. In this task, you are to apply PCA to reduce the dimension of features.

### Deliverables

- Code implementation of PCA on the train and test sets. Note that you are allowed to use the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) package for this task.
- Report the Macro F1 scores for applying 2000, 1000, 500, and 100 components on the test set. Note that you will have to submit your predicted labels to Kaggle to retrieve the Macro F1 scores for the test set and report the results in your final report. Use KNN as the machine learning model for your training and prediction (You are also allowed to use the [sklearn package for KNN implementation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) (set n_neighbors=2).

## Task 3: Try Other Methods (Open-Ended)

In this course, you are exposed to many other machine learning models. For this task, you can apply any other machine learning models (taught in the course or not) to improve the hate speech classification performance! Nevertheless, you are NOT TO use any deep learning approach (if you are keen on deep learning, please sign up for the Deep Learning course! - highly encourage!).

To make this task fun, we will have a race to the top! Bonus marks will be awarded as follows:

- 1 mark: For the third-highest score on the private leaderboard.
- 2 marks: For the second-highest score on the private leaderboard.
- 3 marks: For the top-highest score on the private leaderboard.

Note that the private leaderboard will only be released after the project submission. The top 3 teams will present their solution on week 13 to get the bonus marks!

### Deliverables

- Code implementation of all the models that you have tried. Please include comments on your implementation (i.e., tell us the models you have used and list the key hyperparameter settings).
- Submit your predicted labels for the test set to Kaggle. You will be able to see your model performance on the public leaderboard. Please make your submission under your registered team name! We will award the points according to the ranking of the registered team name.
