#!/usr/bin/env python

# Name: Shivansh Rustagi
# CruzID: shrustag
# ID: 1651034
# CSE 143 Assignment 1
import nltk, zipfile, argparse, sys
from contextlib import redirect_stdout

###############################################################################
## Utility Functions ##########################################################
###############################################################################
# This method takes the path to a zip archive.
# It first creates a ZipFile object.
# Using a list comprehension it creates a list where each element contains
# the raw text of the document file.
# We iterate over each named file in the archive:
#     for fn in zip_archive.namelist()
# For each file that ends with '.txt' we open the file in read only
# mode:
#     zip_archive.open(fn, 'rU')
# Finally, we read the raw contents of the file:
#     zip_archive.open(fn, 'rU').read()
def unzip_corpus(input_file):
	zip_archive = zipfile.ZipFile(input_file)
	contents = [zip_archive.open(fn, 'r').read().decode('utf-8') for fn in zip_archive.namelist() if fn.endswith(".txt")
				and not fn.startswith('__MACOSX')]
	return contents


###############################################################################
## Stub Functions #############################################################
###############################################################################
def process_corpus(corpus_name):
	input_file = corpus_name + ".zip"
	corpus_contents = unzip_corpus(input_file)

	##################
	#  Tokenization  #
	##################
	#	Write name of corpus to `stdout`
	print("Corpus name: " + corpus_name, file=sys.stdout)
	
	#	Delimit the sentences for each document in the corpus.
	tokenized_sentences = [nltk.sent_tokenize(sent) for sent in corpus_contents]

	#	Tokenize the words in each sentence of each document.
	#		`tokenized_words[0]` is the first document in the corpus
	#		`tokenized_words[0][0]` is the first sentence of the first document in the corpus
	#		`tokenized_words[0][0][0]` is the first word of the first sentence of the first document in the corpus
	tokenized_words = []
	corpus_length = 0
	for document in tokenized_sentences:
		temp = []
		for sentences in document:
			result = nltk.word_tokenize(sentences)
			temp.append(result)
			corpus_length += len(result)
		tokenized_words.append(temp)
	
	# Count the number of total words in the corpus and write the result to `stdout`.
	#	corpus_length calculated in previous step for convenience
	#	lowercase is irrelevant in this step because we want total tokens in corpus, not unique, so no need to account for duplicates
	print("Total words in the corpus: " + str(corpus_length), file=sys.stdout)

	####################
	#  Part-of-Speech  #
	####################
	#	Apply the default part-of-speech tagger to each tokenized sentence
	# 	Write a file named CORPUS NAME-pos.txt that has each part-of-speech tagged sentence on a separate line and a blank newline separating documents. Where CORPUS NAME is either fables or blogs. The format of the tagging should be a word-tag pair with a / in between. For example: The/DT boy/NN jumped/VBD ./.
	filename = corpus_name + "-pos.txt"
	with open(filename, "w+") as f:
		for document in tokenized_words:
			for sentence in document:
				result = nltk.pos_tag(sentence)
				for token in result:
					print((token[0] + "/" + token[1]), file=f, end=' ')
			print('\n', file=f)

	###############
	#  Frequency  #
	###############
	# Write the vocabulary size of corpus to `stdout`. Please note that you should use the lowercased word
	vocabulary  = []
	for document in tokenized_sentences:
		for sentence in document:
			vocabulary.extend(nltk.word_tokenize(sentence.lower()))
	print("Vocabulary size of the corpus: " + str(len(set(vocabulary))), file=sys.stdout)

	# Write the most frequent part-of-speech tag and its frequency to the stdout.
	dist = nltk.FreqDist(tag for (word, tag) in nltk.pos_tag(vocabulary))
	print("The most frequent part-of-speech tag is " + dist.most_common(1)[0][0] + " with frequency " + str(dist.most_common(1)[0][1]), file=sys.stdout)

	# Write down the top 10 most frequent part-of-speech tags in the corpus with their frequency and relative frequency to the stdout in a decreasing order of frequency. Relative frequency should be rounded to 3 digits and can be computed by dividing the frequency by the total number of tokens in the corpus.
	print("Frequencies and relative frequencies of the top ten part-of-speech tags in the corpus in decreasing order of frequency are: ", end='', file=sys.stdout)
	for result in dist.most_common(9):
		print(result[0] + " has frequency " + str(result[1]) + " and relative frequency " + "{:.3e}".format(round(result[1]/corpus_length, 3)), file=sys.stdout, end=', ')
	print('and finally, ' + dist.most_common(10)[9][0] + " has frequency " + str(dist.most_common(10)[9][1]) + " and relative frequency " + "{:.2e}".format(round(dist.most_common(10)[9][1]/corpus_length, 3)), file=sys.stdout)

	# Find the frequency of each unique word (after lowercasing) using the FreqDist module and write the list in decreasing order to a file named CORPUS NAME-word-freq.txt.
	dist = nltk.FreqDist(vocabulary)
	filename = corpus_name + "-word-freq.txt"
	with open(filename, "w+") as f:
		for token in list(dist.most_common()):	
			print(token[0] + " has frequency " + str(token[1]), file=f)

	# Find the frequency of each word given its part-of-speech tag. Use a conditional frequency distribution for this (CondFreqDist) where the first item in the pair is the part-of-speech and the second item is the lowercased word. Note, the part-of-speech tagger requires uppercase words and returns the word/tag pair in the inverse order of what we are asking here. Use the tabulate() method of the CondFreqDist class to write the results to a file named CORPUS NAME-pos-word-freq.txt.
	part_of_speeches = nltk.pos_tag(vocabulary)
	dist = nltk.ConditionalFreqDist((tag, word) for (word, tag) in part_of_speeches)
	filename = corpus_name + "-pos-word-freq.txt"
	with open(filename, 'w+') as f:
	    with redirect_stdout(f):
	        dist.tabulate()

	###################
	#  Similar Words  #
	###################
	# For the most frequent word in the NN (nouns), VBD (past-tense verbs), JJ (adjectives) and RB (adverbs) part-of-speech tags, find the most similar words using Text.similar(). Write the output to stdout (this will happen by default).
	noun_most_freq = dist['NN'].most_common(1)[0][0]
	past_tense_verb_most_freq = dist['VBD'].most_common(1)[0][0]
	adjective_most_freq = dist['JJ'].most_common(1)[0][0]
	adverb_most_freq = dist['RB'].most_common(1)[0][0]
	corpus_text = nltk.Text(vocabulary)
	print("The most frequent word in the NN category is: \'" + noun_most_freq + "\' and its similar words are: ", end='')
	corpus_text.similar(noun_most_freq)
	print("The most frequent word in the VBD category is: \'" + past_tense_verb_most_freq + "\' and its similar words are: ", end='')
	corpus_text.similar(past_tense_verb_most_freq)
	print("The most frequent word in the JJ category is: \'" + adjective_most_freq + "\' and its similar words are: ", end='')
	corpus_text.similar(adjective_most_freq)
	print("The most frequent word in the RB category is: \'" + adverb_most_freq + "\' and its similar words are: ", end='')
	corpus_text.similar(adverb_most_freq)

	##################
	#  Collocations  #
	##################
	# Write the collocations to the stdout.
	print("Collocations: " + ', '.join(corpus_text.collocation_list()), file=sys.stdout)


###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Assignment 1')
	parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',  help='Which corpus to process {documents, blogs}')

	args = parser.parse_args()
	
	corpus_name = args.corpus
	
	if corpus_name == "documents" or "blogs":
		process_corpus(corpus_name)
	else:
		print("Unknown corpus name: {0}".format(corpus_name))
	