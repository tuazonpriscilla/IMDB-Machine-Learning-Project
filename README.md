# Project Description
In this project I used IMDB data to train a machine learning algorithm to be able to recognize whether a review is positive or negative depending on the words used. The IMDB data consists of movie 
reviews and their sentiment, whether the movie review was positive or negative. This kind of machine learning model could be used by production companies to see whether their movies are getting more positive or 
negative reviews.

# Data Preparation 
To prepare my data I cleaned the reviews by making all the letters lowercase and removing stopwords. I then separated the data into training and testing sets. I then transformed these strings of reviews into word
count vectors and used PCA to extract the information that was most useful. 

# Metrics
Since this is a classification problem I used accuracy and confusion matrices as my metrics. These are the best metrics for my model since accuracy showed me the percentage of correct classificaitons
my algorithm is making and the confusion matrices showed me what the algorithms were getting wrong and what they were getting right.

# Model Planning and Execution
For this project I used a SVC, perceptron, and a MLP as my learning algorithms.

For my SVC I started with a baseline then tuned it. I tuned it by changing break_ties to True and changing my cache size to 300.

For my perceptron I started with a baseline then tuned it. I tuned it by increasing the alpha, the max number of iterations, and changing the early_stopping value to True.

For my MLP I started with a baseline then tuned it. I tuned it by changing the learning rate to adaptive and increased the hidden layer size to 150.

# Results
All of my accuracies for my algorithms were pretty close, the difference between the lowest and the highest accuracy being about 8%. My tuned MLP performed the best at about 81%. After that my baseline MLP did 
second best then my tuned perceptron, baseline percerptron, tuned SVC, and my baseline SVC did the worst. All of my algorithms had more false postives than false negatives, my MLP having the smallest gap between 
the two.

# Conclusions
I was surprised by how well my SVC performed compared to my other algorithms considering it had much lower numbers for its vectorizer and PCA. I found it interesting how there were so many false positives in my 
results. Maybe the algorithms could be missing the context of the review and classifying certain words incorrectly like on reviews with sarcasm or something of the like. I think that all of my models did 
relatively well considering they all got above 70% but my tuned MLP did a bit better than the rest. It makes sense that the MLP would do better though considering it is many perceptrons and it had more words to 
work with than the SVC. I think that the turned MLP performs well enough to use in real life if a production company were trying to see how well their movies do. It would give them a general idea of the seniment
of reviews and they would still be able to check other sources in case the algorithm falsely gave them too positive of feedback compared to the actual reviews.

