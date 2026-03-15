# ml-three-body-problem
This repository contains the first project for Machine Learning course at NOVA FCT.

Authors:
- Weronika Łoś
- Paweł Spychała
- Piotr Ratajczak

> [!NOTE]
> `X_train` and `X_test` should be in folder `data/`

Here's a rundown of the project from Prof. Claudia Soares's website:

## The Three-Body Problem

<img alt="image" src="three-body.gif"/>

*Twenty examples of special periodic solutions to the three-body problem. By Perosello - Uploaded by Author, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=133294338*
<br><br>
The Three-Body Problem involves determining the motion or trajectory of three bodies, given their initial positions and velocities, in accordance with Newton's laws of motion. Why is this challenging? Let's first consider the two-body problem.

### The Two-Body Problem

The two-body problem is a mathematical problem of how two bodies move over time, given their mass, current speed, and motion direction. It is assumed that both objects move only by gravity.
This problem asks, "What are the trajectories of two masses in space if the only force acting on them is their mutual attraction?" Essentially, we consider an isolated system of two masses and aim to find their trajectories from a given initial position governed by Newton's laws. We can solve this problem by addressing the corresponding system of differential equations. A closed-form solution is possible for two bodies by re-formulating it as two one-body problems, allowing us to obtain explicit expressions for the positions of both bodies over time.

### The Three-Body Problem

The three-body problem extends the two-body problem by adding a third mass. This addition drastically increases the problem's complexity, generally leading to a chaotic, non-periodic system. Explicit solutions for the differential equations of the three-body problem do not exist. Consequently, the problem is typically tackled with numerical integration, which is computationally expensive and time-consuming. Despite long-standing interest, solving this problem remains a significant challenge.

## Tasks

Your goal in this challenge is to learn how to predict the movement of the three bodies on a 2D plane given a set of initial positions, without using numerical solvers. Note that at the initial positions, velocities are zero.
You will have access to a dataset from simulations of the 3-Body Problem under specific initial conditions. You can access position and velocity measurements for each body over several time steps. We denote the position of object 1 at a given time as $(x_1, y_1)$ and the velocity as $(v_{x_1}, v_{y_1})$.

### Task 1: Setting the Baseline

In ML, it is essential to establish a baseline model for comparison with any developments.

#### Task 1.1 Data Preparation and Validation Pipeline
First, explore the training data for any anomalies and visualize some trajectories. Each trajectory consists of 258 lines, one per data point. Note that after a collision, all lines of that trajectory will be zero up to the 258th data point for that trajectory.
You will find more details about the datasets in the Data section of the Kaggle competition page.
You will then split the dataset into training, validation, and test sets to assess the quality of your submissions. Note that these sets should be completely independent to provide an unbiased estimate of the true error when submitting a solution. To ensure this, make sure you do not have the same initial positions in any two sets. Describe your validation strategy on the slides. You might need to code a new train_test_split function.

#### Task 1.2 Learn the baseline model
Learn a baseline model that can predict the position of each of the 3 bodies at a given time t, given a set of initial conditions. Your baseline will be a Linear Regression model.
For the baseline model, make a pipeline and add a StandardScaler instance before the regressor. See the pipeline tutorial on the Tutorials document for the course.
To assess the quality of your model, build the y-y hat plot and examine it. Below, you have a suggestion on how to code this plot.
Submit the result of your baseline model to Kaggle with a submission named baseline-model.csv.

    import matplotlib.pyplot as plt

    def plot_y_yhat(y_test,y_pred, plot_title = "plot"):
        labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
        MAX = 500
        if len(y_test) > MAX:
            idx = np.random.choice(len(y_test),MAX, replace=False)
        else:
            idx = np.arange(len(y_test))
        plt.figure(figsize=(10,10))
        for i in range(6):
            x0 = np.min(y_test[idx,i])
            x1 = np.max(y_test[idx,i])
            plt.subplot(3,2,i+1)
            plt.scatter(y_test[idx,i],y_pred[idx,i])
            plt.xlabel('True '+labels[i])
            plt.ylabel('Predicted '+labels[i])
            plt.plot([x0,x1],[x0,x1],color='red')
            plt.axis('square')
        plt.savefig(plot_title+'.pdf')
        plt.show()

Note, you can save a plot using matplotlib. With the figure open, evaluate the following:

    plt.savefig('baseline.pdf')

Example of y-y hat plot for a baseline model.

Show your plots and write your analysis in the slides, including any large discrepancies between the RMSE you estimated locally from your test split and the one computed by Kaggle.
If there is such a discrepancy, work on your validation strategy and correct it. Update the slides accordingly to describe the issue.

### Task 2: Nonlinear models on the data — the Polynomial model

#### Task 2.1 Development
Develop a function to validate a polynomial regression model with the following signature:

    def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None, degrees=range(1,15), max_features=None)

It should return the best model and the best RMSE.
To run multiple tests quickly, sample a small percentage of the original data for training. Note that the training time increases superlinearly with the polynomial degree.

Print the number of features generated by PolynomialFeatures by checking the attribute n_output_features_ after fitting the pipeline.
Comment on this number in the slides. Whenever you submit to Kaggle, train with the best-performing degree from the training dataset — and only that one degree, to avoid feature complexity.
Also, check using RidgeCV and other regularization methods.
Run your function ten times and examine the distribution of the selected polynomial degrees. Show a plot of this distribution on the slides. Comment on your findings and select the best degree from this analysis.

#### Task 2.2 Evaluation
Evaluate your new nonlinear model against your baseline. Compare RMSE and the y-y hat plots.
Comment on the results.
Submit to Kaggle with the file name polynomial_submission.csv.

### Task 3: Feature Engineering

Take the baseline model and explore the possibility of reducing or adding features.
Remember the slides on Feature Engineering and ML in practice.

#### Task 3.1 Removing variables
Explore the features and verify if there are any linear relationships between them.
Remember, features should be uncorrelated with each other and highly correlated with the target variables. Use the Seaborn package to inspect these relationships.

Assume you have your dataset in the df data frame.

    import seaborn as sns
    sns.pairplot(df.sample(200), kind="hist")

The result will be similar to this:
pairplot.pdf

Check also linear correlations:

    corr = df.corr()
    sns.heatmap(corr,annot=True)

Sort the interactions by the absolute value of the correlations.
Eliminate the more redundant variables one by one, and check the impact of eliminating each variable on the baseline model performance using RMSE and y-y hat plots.
Select some of the experiments, add the plots to the slides, and use them to justify your choices for eliminating variables.

#### Task 3.2 Evaluation of Variable Reduction
After deciding on variable reduction, run your validate_poly_regression function on the reduced feature set. Compare the performance with the model from Task 2 using RMSE and the y-y hat plot. Add the information and your comments to the slides. Submit to Kaggle with the name reduced_polynomial_submission.csv.

#### Task 3.3 Adding Variables
Examine your feature pairplot attentively. Try creating new features, like inverses of quantities, norms, ratios, etc. Refer to the Wikipedia page of the three-body problem for other computations that may be suitable candidates. Test the new feature sets with the baseline. Document the evidence you obtained from these experiments, and write your conclusions, always backed by evidence, on your slides.

#### Task 3.4 Evaluation of Variable Augmentation
After analyzing which variables to add to your model, refactor your validate_poly_regression function to include a ColumnTransformer and possibly a FunctionTransformer to implement your augmentation techniques. Be careful with the degree you choose, as polynomial features grow very fast with the degree of the polynomial. Compare the performance with the models from Task 2 and Task 3.2, using RMSE and the y-y hat plot. Add the information and your comments to the slides. Submit to Kaggle with the name augmented_polynomial_submission.csv.

### Task 4: Nonparametric Model — the k-Nearest Neighbors Regressor

#### Task 4.1 Development
Develop a function to validate a kNN regression model with the following signature:

    def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1,15))

How does your training/inference times vary with $k$? Report on the slides showing evidence of the behavior you are studying. Which $k$ gives you better error? Try out the Feature Engineering sets from Task 3. Show evidence and report on the slides.

#### Task 4.2 Evaluation
Evaluate your new nonparametric model against your baseline and the best models in Tasks 2 and 3 by comparing RMSE and y-y hat plots. Add the information and your comments to the slides. Submit to Kaggle with the name knn_submission.csv.

### Task 5 [Optional]

To build your final model, use anything you have learned in the ML course so far.
Submit the predictions to Kaggle and describe your model architecture and options in the slides.
