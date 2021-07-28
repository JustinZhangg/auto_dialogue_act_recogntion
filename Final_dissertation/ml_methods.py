from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn import tree, svm, naive_bayes, neighbors, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

import utils


class MachineLearningClassification(object):
    def __init__(self, filename):
        self.models = OrderedDict([
            ('KNN', neighbors.KNeighborsClassifier(n_neighbors=5,
                                                   weights='uniform',
                                                   algorithm='ball_tree',
                                                   leaf_size=30,
                                                   p=2,
                                                   metric='euclidean',
                                                   metric_params=None,
                                                   n_jobs=-1)),
            #('KNN', neighbors.KNeighborsClassifier()),
            ('Logistic Regression', linear_model.LogisticRegression(C=0.2,  # regularzation
                                                                    class_weight=None,  # 
                                                                    dual=False,  
                                                                    fit_intercept=True,
                                                                    intercept_scaling=1,  
                                                                    max_iter=1000,  
                                                                    multi_class='ovr',  
                                                                    n_jobs=4,  
                                                                    penalty='l2',  # l2
                                                                    random_state=23,
                                                                    solver='sag',  
                                                                    tol=0.0001,  
                                                                    verbose=0,  
                                                                    warm_start=False  
                                                                    )),
            #('Logistic Regression', linear_model.LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=5000)),
            ('SVM', svm.SVC(kernel='linear', max_iter=100)),
            ('Naive Bayes', naive_bayes.MultinomialNB()),
            ('Decision', tree.DecisionTreeClassifier()),
            ('Random Tree',
             RandomForestClassifier(n_estimators=10, max_features=0.05, oob_score=True, n_jobs=-1, random_state=50,
                                    min_samples_leaf=50))

        ])

        self.pipelines = []
        self.texts, self.labels = utils.load_dataset(filename)

        self.process_label_and_data()

    def process_label_and_data(self):
        self.pipelines.append(('vect',
                               CountVectorizer(
                                   lowercase=True,
                                   analyzer='char_wb',
                                   ngram_range=(1, 4),
                                   max_features=1000,
                               )))
        self.pipelines.append(('tfidf', TfidfTransformer()))

    def run(self, model_name: str):
        if model_name not in self.models:
            print("Model Name Is Not Exists !")
        clf = self.models.get(model_name)
        pipeline = []
        pipeline.extend(self.pipelines)
        pipeline.append(('clf', clf))

        print("=====================")
        print("Starting To Run Classification " + model_name)
        mean = self.eval(Pipeline(pipeline))
        print("End Running Classification " + model_name)
        return mean

    def eval(self, classifier):
        """Testing Model"""
        accuracies = cross_val_score(classifier, self.texts, y=self.labels, scoring=None, cv=3, n_jobs=1)
        print('cross validation result:', accuracies)

        plt.plot(range(1, 4), accuracies)
        plt.xlabel('K')
        plt.ylabel('Accuracy')  
        plt.show()
        return accuracies.mean()
