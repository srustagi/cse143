
import re, nltk, pickle, argparse, pprint
import os
import data_helper
from features import get_features_category_tuples

DATA_DIR = "data"

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
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None):
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    probability = []
    for pdist in classifier.prob_classify_many([pair[0] for pair in features_category_tuples]):
        probs = {}
        probs["positive"] = pdist.prob('positive')
        probs["negative"] = pdist.prob('negative')
        probability.append(probs)
    reference_labels = [sample[1] for sample in features_category_tuples]
    features_only = [sample[0] for sample in features_category_tuples]
    predictions = classifier.classify_many(features_only)
    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predictions)
    return accuracy, probability, confusion_matrix


def build_features(data_file, feat_name, save_feats=None, binning=False):
    # read text data
    raw_data = open(os.path.join(DATA_DIR, data_file), "r").read()
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)

    return features_category_tuples, texts


def train_model(datafile, feature_set, save_model=None):
    features_data, texts = build_features(datafile, feature_set)

    classifier = nltk.classify.NaiveBayesClassifier.train(features_data)

    if save_model is not None:
        save_classifier(classifier, save_model)
    return classifier


def train_eval(train_file, feature_set, classifier_fname, eval_file=None):

    # train the model
    split_name = "train"
    model = train_model(train_file, feature_set, classifier_fname)
    model.show_most_informative_features(20)

    # save the model
    if model is None:
        model = get_classifier(classifier_fname)

    # evaluate the model
    if eval_file is not None:
        features_data, texts = build_features(eval_file, feature_set, binning=None)
        accuracy, probability, cm = evaluate(model, features_data, texts, data_set_name=None)
        print("The accuracy of {} is: {}".format(eval_file, accuracy))
        print("Proabability per class:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(probability)
        print("Confusion Matrix:")
        print(str(cm))
    else:
        accuracy = None
    return accuracy


def main():
    # add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 3')
    parser.add_argument('-d', dest="data_fname", default="imdb-training.data", help='File name of the testing data.')
    args = parser.parse_args()

    train_data = args.data_fname
    eval_data = "imdb-testing.data"

    for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features", "word_pos_opinion_features"]:
        print("\nTraining with {}".format(feat_set))
        acc = train_eval(train_data, feat_set, ("imdb-" + feat_set + "-model-P1.pickle"), eval_file=eval_data)


if __name__ == "__main__":
    main()
