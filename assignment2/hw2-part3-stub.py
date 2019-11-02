from collections import Counter
import re, nltk, argparse

def get_score(review):
	return int(re.search(r'Overall = ([1-5])', review).group(1))

def get_text(review):
	return re.search(r'Text = "(.*)"', review).group(1)

def read_reviews(file_name):
	"""
	Dont change this function.

	:param file_name:
	:return:
	"""
	file = open(file_name, "rb")
	raw_data = file.read().decode("latin1")
	file.close()

	positive_texts = []
	negative_texts = []
	first_sent = None
	for review in re.split(r'\.\n', raw_data):
		overall_score = get_score(review)
		review_text = get_text(review)
		if overall_score > 3:
			positive_texts.append(review_text)
		elif overall_score < 3:
			negative_texts.append(review_text)
		if first_sent == None:
			sent = nltk.sent_tokenize(review_text)
			if (len(sent) > 0):
				first_sent = sent[0]
	return positive_texts, negative_texts, first_sent


########################################################################
## Dont change the code above here
######################################################################



def process_reviews(file_name):
	positive_texts, negative_texts, first_sent = read_reviews(file_name)
	# There are 150 positive reviews and 150 negative reviews.
	# print(len(positive_texts))
	# print(len(negative_texts))

	# Your code goes here
	positive_texts = [sent.lower() for sent in positive_texts]
	negative_texts = [sent.lower() for sent in negative_texts]

	clean_positives = []
	clean_negatives = []

	positive_unigrams = []
	negative_unigrams = []

	positive_bigrams = []
	negative_bigrams = []

	positive_vocabulary = []
	negative_vocabulary = []

	for review in positive_texts:
		stop_words = set(nltk.corpus.stopwords.words('english')) 
		word_tokens = nltk.word_tokenize(review)
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		banned_content = set([re.findall(r"[^\w]*", word)[0] for word in filtered_sentence if re.findall(r"[^\w]", word) != []])
		filtered_sentence = [w for w in filtered_sentence if not w in banned_content]
		positive_vocabulary.extend(filtered_sentence)
		positive_unigrams.extend([word[0] for word in list(nltk.ngrams(filtered_sentence, 1))])
		positive_bigrams.extend(list(nltk.bigrams(filtered_sentence)))
		clean_positives.append(" ".join(filtered_sentence))

	for review in negative_texts:
		stop_words = set(nltk.corpus.stopwords.words('english')) 
		word_tokens = nltk.word_tokenize(review)
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		banned_content = set([re.findall(r"[^\w]*", word)[0] for word in filtered_sentence if re.findall(r"[^\w]", word) != []])
		filtered_sentence = [w for w in filtered_sentence if not w in banned_content]
		negative_vocabulary.extend(filtered_sentence)
		negative_unigrams.extend([word[0] for word in list(nltk.ngrams(filtered_sentence, 1))])
		negative_bigrams.extend(list(nltk.bigrams(filtered_sentence)))
		clean_negatives.append(" ".join(filtered_sentence))

	pos_unigram_dist = nltk.FreqDist(positive_unigrams)
	neg_unigram_dist = nltk.FreqDist(negative_unigrams)

	with open("positive-unigram-freq.txt", "w+", encoding="utf-8") as f:
		for token in list(pos_unigram_dist.most_common()):	
			print(token[0] + " " + str(token[1]), file=f)

	with open("negative-unigram-freq.txt", "w+", encoding="utf-8") as f:
		for token in list(neg_unigram_dist.most_common()):	
			print(token[0] + " " + str(token[1]), file=f)

	pos_bigram_dist = nltk.ConditionalFreqDist((condition_word, word) for (condition_word, word) in positive_bigrams)
	neg_bigram_dist = nltk.ConditionalFreqDist((condition_word, word) for (condition_word, word) in negative_bigrams)
	
	with open("positive-bigram-freq.txt", 'w+', encoding="utf-8") as f:
		for condition in pos_bigram_dist.conditions():
			for (word, freq) in pos_bigram_dist[condition].most_common():
				print(condition, word, freq, file=f)
	
	with open("negative-bigram-freq.txt", 'w+', encoding="utf-8") as f:
		for condition in neg_bigram_dist.conditions():
			for (word, freq) in neg_bigram_dist[condition].most_common():
				print(condition, word, freq, file=f)

	pos_unigram_counts = Counter(nltk.ngrams(positive_vocabulary, 1)).most_common(5)
	pos_bigram_counts = Counter(nltk.ngrams(positive_vocabulary, 2)).most_common(5)
	pos_trigram_counts = Counter(nltk.ngrams(positive_vocabulary, 3)).most_common(5)
	pos_fourgram_counts = Counter(nltk.ngrams(positive_vocabulary, 4)).most_common(5)
	pos_fivegram_counts = Counter(nltk.ngrams(positive_vocabulary, 5)).most_common(5)

	# print("\n", pos_unigram_counts, "\n", pos_bigram_counts, "\n", pos_trigram_counts, "\n", pos_fourgram_counts, "\n", pos_fivegram_counts)

	neg_unigram_counts = Counter(nltk.ngrams(negative_vocabulary, 1)).most_common(5)
	neg_bigram_counts = Counter(nltk.ngrams(negative_vocabulary, 2)).most_common(5)
	neg_trigram_counts = Counter(nltk.ngrams(negative_vocabulary, 3)).most_common(5)
	neg_fourgram_counts = Counter(nltk.ngrams(negative_vocabulary, 4)).most_common(5)
	neg_fivegram_counts = Counter(nltk.ngrams(negative_vocabulary, 5)).most_common(5)

	# print("\n", neg_unigram_counts, "\n", neg_bigram_counts, "\n", neg_trigram_counts, "\n", neg_fourgram_counts, "\n", neg_fivegram_counts)

	positive_vocabulary = nltk.Text(positive_vocabulary)
	negative_vocabulary = nltk.Text(negative_vocabulary)

	print("Positive Reviews Collocations: " + ', '.join(positive_vocabulary.collocation_list()))
	print("Negative Reviews Collocations: " + ', '.join(negative_vocabulary.collocation_list()))


# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
	file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
	file.write(data)
	file.close()


def write_unigram_freq(category, unigrams):
	"""
	A function to write the unigrams and their frequencies to file.

	:param category: [string]
	:param unigrams: list of (word, frequency) tuples
	:return:
	"""
	uni_file = open("{0}-unigram-freq-n.txt".format(category), 'w', encoding="utf-8")
	for word, count in unigrams:
		uni_file.write("{0:<20s}{1:<d}\n".format(word, count))
	uni_file.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Assignment 2')
	parser.add_argument('-f', dest="fname", default="restaurant-training.data",  help='File name.')
	args = parser.parse_args()
	fname = args.fname

	process_reviews(fname)
