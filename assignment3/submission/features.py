import nltk
import re
import word_category_counter
import data_helper
import os, sys

DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)

def normalize(token, should_normalize=True):
	"""
	This function performs text normalization.

	If should_normalize is False then we return the original token unchanged.
	Otherwise, we return a normalized version of the token, or None.

	For some tokens (like stopwords) we might not want to keep the token. In
	this case we return None.

	:param token: str: the word to normalize
	:param should_normalize: bool
	:return: None or str
	"""
	if not should_normalize:
		normalized_token = token

	else:
		stop_words = set(nltk.corpus.stopwords.words('english'))
		normalized_token = token.lower()
		if normalized_token in stop_words or re.search(r"\w+", normalized_token) == None:
			return None
	return normalized_token


def get_words_tags(text, should_normalize=True):
	"""
	This function performs part of speech tagging and extracts the words
	from the review text.

	You need to :
		- tokenize the text into sentences
		- word tokenize each sentence
		- part of speech tag the words of each sentence

	Return a list containing all the words of the review and another list
	containing all the part-of-speech tags for those words.

	:param text:
	:param should_normalize:
	:return:
	"""
	words = []
	tags = []

	sentences = nltk.sent_tokenize(text)
	words.extend(nltk.word_tokenize(text))
	tags = [tag for (word, tag) in nltk.pos_tag(words)]
	return words, tags


def get_ngram_features(tokens):
	"""
	This function creates the unigram and bigram features as described in
	the assignment3 handout.

	:param tokens:
	:return: feature_vectors: a dictionary values for each ngram feature
	"""
	feature_vectors = {}
	tokens = [normalize(token) for token in tokens if normalize(token) != None]
	
	token_unigrams = list(nltk.ngrams(tokens, 1))
	token_bigrams = list(nltk.ngrams(tokens, 2))
	token_trigrams = list(nltk.ngrams(tokens, 3))
	
	num_uni = len(token_unigrams)
	num_bi = len(token_bigrams)
	num_tri = len(token_trigrams)

	uni_dist = nltk.FreqDist(token_unigrams)
	bi_dist = nltk.FreqDist(token_bigrams)
	tri_dist = nltk.FreqDist(token_trigrams)

	for sample in uni_dist.most_common():
		sample_key = "UNI_" + sample[0][0]
		sample_value = sample[1] / num_uni
		feature_vectors[sample_key] = sample_value

	for sample in bi_dist.most_common():
		sample_key = "BIGRAM_" + "_".join(sample[0])
		sample_value = sample[1] / num_bi
		feature_vectors[sample_key] = sample_value

	for sample in tri_dist.most_common():
		sample_key = "TRIGRAM_" + "_".join(sample[0])
		sample_value = sample[1] / num_tri
		feature_vectors[sample_key] = sample_value

	return feature_vectors


def get_pos_features(tags):
	"""
	This function creates the unigram and bigram part-of-speech features
	as described in the assignment3 handout.

	:param tags: list of POS tags
	:return: feature_vectors: a dictionary values for each ngram-pos feature
	"""
	feature_vectors = {}
	
	tags = [normalize(tag) for tag in tags if normalize(tag) != None]

	tag_unigrams = list(nltk.ngrams(tags, 1))
	tag_bigrams = list(nltk.ngrams(tags, 2))
	tag_trigrams = list(nltk.ngrams(tags, 3))
	
	num_uni = len(tag_unigrams)
	num_bi = len(tag_bigrams)
	num_tri = len(tag_trigrams)

	uni_dist = nltk.FreqDist(tag_unigrams)
	bi_dist = nltk.FreqDist(tag_bigrams)
	tri_dist = nltk.FreqDist(tag_trigrams)

	for sample in uni_dist.most_common():
		sample_key = "UNI_POS_" + (sample[0][0]).upper()
		sample_value = sample[1] / num_uni
		feature_vectors[sample_key] = sample_value

	for sample in bi_dist.most_common():
		sample_key = "BIGRAM_POS_" + ("_".join(sample[0])).upper()
		sample_value = sample[1] / num_bi
		feature_vectors[sample_key] = sample_value

	for sample in tri_dist.most_common():
		sample_key = "TRIGRAM_POS_" + ("_".join(sample[0])).upper()
		sample_value = sample[1] / num_tri
		feature_vectors[sample_key] = sample_value

	return feature_vectors


def get_liwc_features(words):
	"""
	Adds a simple LIWC derived feature

	:param words:
	:return:
	"""

	# TODO: binning

	feature_vectors = {}
	text = " ".join(words)
	liwc_scores = word_category_counter.score_text(text)

	# All possible keys to the scores start on line 269
	# of the word_category_counter.py script
	negative_score = liwc_scores["Negative Emotion"]
	positive_score = liwc_scores["Positive Emotion"]
	feature_vectors["Negative Emotion"] = negative_score
	feature_vectors["Positive Emotion"] = positive_score

	feature_vectors["LIWC_OPTIMISM"] = liwc_scores["Optimism and energy"]
	# feature_vectors["LIWC_PAST_TENSE"] = liwc_scores["Past Tense"]
	feature_vectors["LIWC_PEOPLE"] = liwc_scores["Other references to people"]
	feature_vectors["LIWC_HUMANS"] = liwc_scores["Humans"]
	feature_vectors["LIWC_COMMUNINCATION"] = liwc_scores["Communication"]
	# feature_vectors["LIWC_SEE"] = liwc_scores["See"]
	feature_vectors["LIWC_NEGATIONS"] = liwc_scores["Negations"]


	if positive_score > negative_score:
		feature_vectors["liwc:positive"] = 1
	else:
		feature_vectors["liwc:negative"] = 1

	return feature_vectors


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features"}


def get_opinion_features(tags):
	"""
	This function creates the opinion lexicon features
	as described in the assignment3 handout.

	the negative and positive data has been read into the following lists:
	* neg_opinion
	* pos_opinion

	if you haven't downloaded the opinion lexicon, run the following commands:
	*  import nltk
	*  nltk.download('opinion_lexicon')

	:param tags: tokens
	:return: feature_vectors: a dictionary values for each opinion feature
	"""
	neg_opinion = nltk.corpus.opinion_lexicon.negative()
	pos_opinion = nltk.corpus.opinion_lexicon.positive()
	feature_vectors = {}

	# feature_vectors["OPINION_NEG"] = 0
	# feature_vectors["OPINION_POS"] = 0

	normalized_token = [normalize(tkn) for tkn in tags]

	for token in neg_opinion:
		if token in normalized_token and not token in pos_opinion:
			opinion_key = "OPINION_NEG_" + token.upper()
			feature_vectors[opinion_key] = 1
		else:
			opinion_key = "OPINION_NEG_" + token.upper()
			feature_vectors[opinion_key] = 0
		
	for token in pos_opinion:
		if token in normalized_token and not token in neg_opinion:
			opinion_key = "OPINION_POS_" + token.upper()
			feature_vectors[opinion_key] = 1
		else:
			opinion_key = "OPINION_POS_" + token.upper()
			feature_vectors[opinion_key] = 0

	return feature_vectors


def get_features_category_tuples(category_text_dict, feature_set):
	"""

	You will might want to update the code here for the competition part.

	:param category_text_dict:
	:param feature_set:
	:return:
	"""
	features_category_tuples = []
	all_texts = []

	assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

	for category in category_text_dict:
		for text in category_text_dict[category]:

			words, tags = get_words_tags(text)
			feature_vectors = {}

			if feature_set == "word_features":
				feature_vectors.update(get_ngram_features(words))
			elif feature_set == "word_pos_features":
				feature_vectors.update(get_ngram_features(words))
				feature_vectors.update(get_pos_features(tags))
			elif feature_set == "word_pos_liwc_features":
				feature_vectors.update(get_ngram_features(words))
				feature_vectors.update(get_pos_features(tags))
				feature_vectors.update(get_liwc_features(words))
			elif feature_set == "word_pos_opinion_features":
				feature_vectors.update(get_ngram_features(words))
				feature_vectors.update(get_pos_features(tags))
				feature_vectors.update(get_liwc_features(words))
				feature_vectors.update(get_opinion_features(words))

			features_category_tuples.append((feature_vectors, category))
			all_texts.append(text)

	return features_category_tuples, all_texts


def write_features_category(features_category_tuples, outfile_name):
	"""
	Save the feature values to file.

	:param features_category_tuples:
	:param outfile_name:
	:return:
	"""
	with open(outfile_name, "w+", encoding="utf-8") as fout:
		for (features, category) in features_category_tuples:
			fout.write("{0:<10s}\t{1}\n".format(category, features))


def features_stub():
	
	feature_sets = ["word_features", "word_pos_features", "word_pos_liwc_features", "word_pos_opinion_features"]
	datasets = ["training", "development", "testing"]
	
	for dataset in datasets:
		for feature_set in feature_sets:
			datafile = "data/imdb-" + dataset + ".data"
			raw_data = data_helper.read_file(datafile)
			positive_texts, negative_texts = data_helper.get_reviews(raw_data)
			category_texts = {"positive": positive_texts, "negative": negative_texts}
			features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)
			filename = "best_features/" + feature_set + "-" + dataset + "-features.txt"
			write_features_category(features_category_tuples, filename)

if __name__ == "__main__":
	features_stub()
