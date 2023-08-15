# Sentiment-Analysis-Model-Comparison
This repository contains 4 different sentiment analysis models and tests their efficiency based on key metrics to draw a conclusion on what is the most optimal common sentiment  analysis model.

#### Step 1: Obtaining the Dataset
The dataset used for all models in this repository is the IMDB movie review dataset in the '''datasets''' package

#### Step 2: Pre-Processing of data
##### REMOVING PUNCTUATIONS
Using regular expression(regex), remove punctuation, hashtags and @-mentions from each tweet.</br>

##### TOKENIZATION
In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization.</br>

##### STEMMING
Stemming is the process of reducing inflected words to their word stem, base, or root form—generally a written word form.</br>

##### LEMMATIZATION
Lemmatization in linguistics is the process of grouping together the inflected forms of a word so they can be analyzed as a single item, identified by the word's lemma, or dictionary form.</br>

#### Step 3: Fitting and training the model
##### Implemented Algorithms
###### NAIVE BAYES CLASSIFIER
###### SENTIMENT INTENSITY ANALYZER
###### LOGISTIC REGRESSION
###### distilBERT
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).
###### SUPPORT VECTOR MACHINE
In machine learning, support-vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.

#### Conclusion
Among all other techniques used, Random Forest Classifier has performed best with the highest accuracy. One reason why RF works well is that the algorithm can look past and handle the missing values in the tweets.

#### Step 4: Model prediction and result comparison

### Limitations of Sentiment Analysis
* One of the downsides of using lexicons is that people express emotions in different ways. Some may express sarcasm and irony in the statements.
* Multilingual sentiment analysis.
* Making the model automatic. Automatic methods, contrary to rule-based systems, don't rely on manually crafted rules, but on machine learning techniques. A sentiment analysis task is usually modeled as a classification problem, whereby a classifier is fed a text and returns a category
* Can take emoticons into account to predict better.
* Apart from the positive and negative categories, the model could be developed to learn to classify tweets that are satirical or sarcastic.
