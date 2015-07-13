import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef, BNode, Literal
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix
from helper.download import ensure_directory
from helper.dataset import Dataset
from helper.text import print_headline


class Prediction:

    def __init__(self, true=None, predicted=None, classes=None):
        self.true = true
        self.predicted = predicted
        self.classes = classes

    def print_scores(self):
        print_headline('Results')
        scores = classification_report(self.true, self.predicted,
            target_names=self.classes)
        print(scores)

    def plot_confusion_matrix(self):
        confusion = confusion_matrix(self.true, self.predicted)
        # Normalize range
        confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        # Create figure
        plt.figure()
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def train_and_predict(classifier, dataset, split, log=True):
    # Convert to numpy arrays and split
    training, testing = dataset.split(split, log)
    # Normalize dataset
    means, stds = training.normalize()
    # TODO: Store params
    testing.normalize(means, stds)
    # Train classifier
    classifier.fit(training.data, training.target)
    # Use model to make predictions
    predicted = classifier.predict(testing.data)
    probabilities = classifier.predict_proba(testing.data)
    prediction = Prediction(testing.target, predicted, testing.classes)
    for index, uri in enumerate(testing.uris):
        subject = URIRef('http://commons.dbpedia.org/resource/File:' + uri.replace('http://commons.wikimedia.org/wiki/Special:FilePath/', ''))
        for classIndex, probability in enumerate(probabilities[index]):
            probability = round(probability * 100) / 100
            g1.add((subject, URIRef('http://commons.dbpedia.org/property/HpiCategory' + testing.classes[classIndex]), Literal(probability)))
            g3bNode = BNode()
            g3.add((g3bNode, URIRef('http://commons.dbpedia.org/property/HpiType'), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
            g3.add((g3bNode, URIRef('http://commons.dbpedia.org/property/HpiProbability'), Literal(probability)))
            if classIndex == (predicted[index]):
                g3.add((subject, URIRef('http://commons.dbpedia.org/property/HpiPrediction'), g3bNode))
                g1.add((subject, URIRef('http://commons.dbpedua.org/property/HpiTopPrediction'), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
                g2.add((subject, URIRef('http://commons.dbpedua.org/property/HpiTopPrediction'), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
            else:
                g3.add((subject, URIRef('http://commons.dbpedia.org/property/HpiCategory'), g3bNode))
            if probability > 0.4:
                g2.add((subject, URIRef('http://commons.dbpedia.org/property/HpiThreshold' + str(0.4)), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
            elif probability > 0.3:
                g2.add((subject, URIRef('http://commons.dbpedia.org/property/HpiThreshold' + str(0.3)), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
            elif probability > 0.2:
                g2.add((subject, URIRef('http://commons.dbpedia.org/property/HpiThreshold' + str(0.2)), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
            elif probability > 0.1:
                g2.add((subject, URIRef('http://commons.dbpedia.org/property/HpiThreshold' + str(0.1)), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
            else:
                g2.add((subject, URIRef('http://commons.dbpedia.org/property/HpiThresholdBelow' + str(0.1) ), URIRef('http://commons.dbpedia.org/resource/Hpi' + testing.classes[classIndex])))
    return prediction

def train_and_predict_Multi_Classifiers(classifiers, end_classifier, dataset, split, log=True):
    training, testing = dataset.split(split, log)
    means, stds = training.normalize()
    testing.normalize(means, stds)
    classifiers[0].fit(training.data, training.target)
    probabilities = classifiers[0].predict_proba(training.data)
    predictions = classifiers[0].predict_proba(testing.data)
    for classifier in classifiers:
        classifier.fit(training.data, training.target)
        probabilities = np.hstack((probabilities, classifier.predict_proba(training.data)))
        predictions = np.hstack((predictions, classifier.predict_proba(testing.data)))
    end_classifier.fit(probabilities, training.target)
    pred = end_classifier.predict(predictions)
    predicted = Prediction(testing.target, pred, testing.classes)
    return predicted


def evaluate_classifier(dataset, classifier, iterations, split):
    scores = []
    for _ in range(iterations):
        prediction = train_and_predict(classifier, dataset, split, log=False)
        score = f1_score(prediction.true, prediction.predicted,
            average='weighted')
        scores.append(score)
    worst = min(scores)
    average = sum(scores) / len(scores)
    best = max(scores)
    return worst, average, best


if __name__ == '__main__':
    parser = ArgumentParser(description='Learning algorithm used to classify \
        images.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('features',
        help='Path to the JSON file containing extracted features of the \
        dataset')
    parser.add_argument('-s', '--split', type=float, default=0.25,
        help='Fraction of data used for validation')
    parser.add_argument('-c', '--copy-predicted',
        default='<folder>/../<folder>-predicted/',
        help='Folder to copy predicted images into; sub directories for all \
        classes are created; <folder> is the directory of the features file')
    args = parser.parse_args()

    if '<folder>' in args.copy_predicted:
        folder = os.path.splitext(args.features)[0]
        args.copy_predicted = args.copy_predicted.replace('<folder>', folder)

    dataset = Dataset()
    g1 = Graph()
    g2 = Graph()
    g3 = Graph()
    dataset.load(args.features)
    classifierList = []

    classifierList += [RandomForestClassifier(n_estimators=300)]
    classifierList += [SVC(kernel='linear', probability=True)]
    classifierList += [DecisionTreeClassifier()]
    DTreeClassifier = LDA()
    classifier = SVC(kernel='linear', probability=True) # RandomForestClassifier(n_estimators=300)

    prediction = train_and_predict(classifier, dataset, args.split)
    #prediction = train_and_predict_Multi_Classifiers(classifierList, DTreeClassifier, dataset, args.split)
    prediction.print_scores()
    prediction.plot_confusion_matrix()
    g1.serialize(destination='data/exampleGraph1', format='n3')
    g2.serialize(destination='data/exampleGraph2', format='n3')
    g3.serialize(destination='data/exampleGraph3', format='n3')
