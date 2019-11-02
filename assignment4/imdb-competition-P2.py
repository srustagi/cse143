from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from nltk.classify import SklearnClassifier
from sklearn.neural_network import multilayer_perceptron

import nltk, pickle, argparse
import os, random
import data_helper
from features import get_features_category_tuples

random.seed(10)
DATA_DIR = "asg4-data/data"


# TODO: You can expand this feature set
CLASSIFIER_SETS = {"nb", "nb_sklearn", "dt", "dt_sklearn", "svm_sklearn", "nn_sklearn"}


def build_classifier(classifier_type):
    """
    Accepted names in CLASSIFIER_SETS
    :param classifier_type:
    :return:
    """
    # TODO: NEWLY ADDED, YOUR CODE GOES HERE, to add more classifiers
    assert classifier_type in CLASSIFIER_SETS, "unrecognized classifier type:{}, Accepted values:{}".format(classifier_type, CLASSIFIER_SETS)

    if classifier_type == "nb":
        cls = nltk.classify.NaiveBayesClassifier
    else:
        raise NotImplemented

    return cls


def write_features_category(features_category_tuples, output_file_name):
    output_file = open("{}-features.txt".format(output_file_name), "w", encoding="utf-8")
    for (features, category) in features_category_tuples:
        output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()


def get_classifier(classifier_fname):
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    # info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    # for feature, n in classifier.most_informative_features(100):
    #     info_file.write("{0}\n".format(feature))
    # info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None):

    ###     YOUR CODE GOES HERE
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)


    accuracy_results_file = open("{}_results.txt".format(data_set_name), 'w', encoding='utf-8')
    accuracy_results_file.write('Results of {}:\n\n'.format(data_set_name))
    accuracy_results_file.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))

    features_only = []
    reference_labels = []
    for feature_vectors, category in features_category_tuples:
        features_only.append(feature_vectors)
        reference_labels.append(category)

    predicted_labels = classifier.classify_many(features_only)

    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)

    accuracy_results_file.write(str(confusion_matrix))
    accuracy_results_file.write('\n\n')
    accuracy_results_file.close()

    return accuracy, confusion_matrix


def build_features(data_file, feat_name, save_feats=None, binning=False):
    # TODO: YOUR CODE GOES HERE: you need to handle if binning=True
    # read text data
    raw_data = data_helper.read_file(os.path.join(DATA_DIR, data_file))
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)

    return features_category_tuples, texts



def train_model(datafile, feature_set, cls_name, save_model=None):

    features_data, texts = build_features(datafile, feature_set)

    classifier = build_classifier(cls_name).train(features_data)
    if save_model is not None:
        save_classifier(classifier, save_model)
        print('saved model {}'.format(save_model))
    return classifier


def train_eval(train_file, feature_set, cls_name, cls_fname, eval_file=None, is_train=True):

    # train or get saved model
    if is_train:
        model = train_model(train_file, feature_set, cls_name)
    else:
        model = get_classifier(cls_fname)


    if model:
        # evaluate the model
        if eval_file is not None:
            features_data, texts = build_features(eval_file, feature_set, binning=False)
            accuracy, cm = evaluate(model, features_data, texts, data_set_name=None)
            print("The accuracy of {} is: {}".format(eval_file, accuracy))
            print("Confusion Matrix:")
            print(str(cm))
        else:
            accuracy = None

        return accuracy
    else:
        print("no model")


if __name__ == '__main__':
    # Add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 4')
    parser.add_argument('-isTrain', dest="is_train", action="store_true", help='Is it for training or testing')
    parser.add_argument('-cls', dest="classifier_type", default="nb",
                        help='The classifier type is used for training')
    parser.add_argument('-train', dest="train_fname", default="imdb-training.data", help='File Name of the Training Data')
    parser.add_argument('-eval', dest="eval_fname", help='File Name for evaluation')
    parser.add_argument('-c', dest="classifier_fname", default="nb-word_features-classifier.pickle",
                        help='File name of the classifier pickle.')
    parser.add_argument('-o', dest="output_fname", default="nb-word_features-test.txt", help='Output file name.')
    parser.add_argument('-f', dest="feature_set", default="word_features",
                        help='Feature set: word_features, word_pos_features, etc')

    args = parser.parse_args()

    is_train = args.is_train
    classifier_type = args.classifier_type
    train_fname = args.train_fname
    eval_fname = args.eval_fname

    classifier_fname = args.classifier_fname
    output_fname = args.output_fname
    feature_set = args.feature_set

    train_eval(train_fname, feature_set, classifier_type, classifier_fname, eval_file=eval_fname, is_train=is_train)






