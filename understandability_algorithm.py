# foundational modules
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import seaborn as sns
sns.set(style="darkgrid")

#NLP
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download("stopwords")
import string

# data preparation for model learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

### classification metrics
from sklearn.metrics import classification_report

class Understandability(object):
    def __init__(self, dataset_path, debug = True):

        # instance vars
        self.path = dataset_path
        self.debug = debug
        self.model = None
        pass

    def _syllables(self, word):
        syllable_count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            syllable_count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                syllable_count += 1
        if word.endswith('e'):
            syllable_count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        if syllable_count == 0:
            syllable_count += 1
        return syllable_count

    def stratified_split(self, X, y,
                         test_size=0.2,
                         validate_size=0.2,
                         random_state=0):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state = random_state)

        # need to calculate new split size.
        # let's assume we had 100 samples and we don't do this
        # then the split will be 20 + (20% of 80) + (80% of 80).
        # But we want 20 + 20 + 60
        new_validate_size = validate_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, stratify=y_train,
            test_size=new_validate_size,
            random_state = random_state)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def train(self):

        df = pd.read_csv("data_to_train.csv")

        # ### Step-2: Split Data into Train, Validation, and Test Data Sets

        y = np.array ( df.level)
        X = np.array ( df.drop ( columns = ['level'] ) )


        # Split data into training, validation, and testing data sets
        X_train, X_test, X_val, y_train, y_test, y_val = self.stratified_split(X, y, random_state = 66)

        if self.debug == True:
            print ("Training (X_train and y_train): \t", X_train.shape, " \t", y_train.shape)
            print ("Validation (X_val and y_val): \t\t", X_val.shape, " \t", y_val.shape)
            print ("Testing (X_test and y_test): \t\t", X_test.shape, "  \t", y_test.shape)

        # Set-up and Build Decision Tree classifier model
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)

        if self.debug == True:
            # Decision Tree model performance
            print("Decision Tree Performance:")
            print ("\tTRAIN Accuracy: {:.2f}".format(dt.score(X_train, y_train)))
            print ("\tVALIDATION Accuracy: {:.2f}".format(dt.score(X_val, y_val)))
            print ("\tTEST Accuracy: {:.2f}".format(dt.score(X_test, y_test)))
            feature_names = df.drop ( columns = ['level']).columns
            fdf = pd.DataFrame(data = list(zip(feature_names, dt.feature_importances_)),
                               columns = ["Feature Names", "Feature Importances"])
            fdf.head(3)
            sns.barplot(y = "Feature Names", x="Feature Importances", data = fdf,
                       color="salmon", saturation=1.0)
            plt.show()

            DT_predictionsValidate = dt.predict(X_val)
            print (classification_report(y_val, DT_predictionsValidate))

        self.model = dt
        return dt

    def predict(self, input_sentence):
        user = input_sentence
        user_string = user.translate(str.maketrans('', '', string.punctuation))
        sent_tokenize(user_string)
        user_words = word_tokenize(user_string)
        user_list = [word for word in user_words if not word in stopwords.words()]
        user_word_count = len(user_list)
        user_char_count = 0
        for word in user_list:
            user_char_count = user_char_count + len(word)
        user_sly_count = 0
        for word in user_list:
            user_sly_count = user_sly_count + self._syllables(word)
        #enter user vocab list here, calculate percentage of words understood
        user_data = [[user_word_count, user_char_count, user_sly_count]]
        X = pd.DataFrame(user_data, columns = ['word_count', 'char_count', 'sly_count'])

        MyPrediction = self.model.predict(X)
        return MyPrediction[0]
