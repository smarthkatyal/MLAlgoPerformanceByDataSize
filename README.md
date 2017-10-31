# MLAlgoPerformanceByDataSize



1. Before executing the task, configure the 'config.py' file as per instructions given in the config file.
2. To execute the code, execute the 'main.py' python file from the console/command prompt.
3. Wait for the output on the console.


Findings/AnswerQuestion
1: To what extent does the effectiveness of machine-learning algorithms depend on the size and complexity of the data? 
Answer::
The effectiveness of machine-learning algorithms depends to a vast extent on the size of the data. From the chart given above, it is observed that the prediction metrics vary with the number of data-points taken into consideration. Although, it cannot be conclusively said that the higher the size of the data, the better the effectiveness of machine-learning algorithms. The chart clearly depicts instances where the size of the dataset has noticeable effect on the effectiveness but there are instances where a larger dataset has resulted in the lowering of effectiveness of the algorithm. The effectiveness of thealgorithm depends on the complexity of the problem i.e.how a dependent variable is related to the independent variable. There needs to be enough data to capture this relationship that may or may not exist between input and output variables. From the above chart, it was also observed that the effectiveness of the algorithm also depends on the complexity of the data. It was observed that for data without noise, the predictions were accurate or close to accurate thus increasing effectiveness. From our observation,if a dataset contains many number of features thatare irrelevant tothe data that is to be predicted, the predictionquality of the model decreases. Hence, simply adding features which are not relevant to the output, will reduce the effectiveness of the algorithm.


2: Looking only at the performance of your best performing algorithm on “The SUM dataset (without noise)”: how well was machine-learning suitable to solve the task of predicting a) the target value and b) the target class? Consider in your assessment, how well a simple rule-based algorithm could have performed.
Answer::
a)Considering the best performing algorithm, in our case Decision Tree Classification, we could predict accurately, all the data in the test sets with 100% accuracy and precision. Hence, we can safely say that machine-learning was suitable approach to predict the data in test set from the given data in the training set.
b)A rule-based algorithm would have performed poorly as compared to Decision Tree Classification. One of the reasons is that the rules may miss scenarios or perform incorrectly due to missing/incorrect values. Machine Learning works on creating a model and few false cases would not affect predictions.Conclusion: We concluded that the effectiveness does depend on the size and complexity of the datain results but are susceptible variousfactors like choice of dataset & choice of algorithm. Both underfitting and overfitting will decrease the performanceof the machine learning algorithms and a right balance should be maintained on deciding the size of the data.Also,the proper selection of dependent variables to determinethe targetvariableaccording to thealgorithmused,plays an important role in determining the results.

Limitation: 1) Fewer data points in a dataset will generally give inappropriate results about the algorithm. 
2) Business scenarios will play a crucial role in deciding the algorithm to be used.
3)Data pre-processing and cleaning is necessary for accurate result prediction.
