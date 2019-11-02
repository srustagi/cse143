import argparse, re, nltk

# https://docs.python.org/3/howto/regex.html
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/


def get_words(pos_sent):
    """
    Given a part-of-speech tagged sentence, return a sentence
    including each token separated by whitespace.

    As an interim step you need to fill word_list with the
    words of the sentence.

    :param pos_sent: [string] The POS tagged stentence
    :return:
    """

    # add the words of the sentence to this list in sequential order.
    word_list = []

    # Your code goes here
    word_list = pos_sent.split(" ")
    word_list = [re.search(r"\w+?(?=\/)", word) for word in word_list]
    word_list = [word.group() for word in word_list if (word != None)]

    # Write a regular expression that matches only the
    # words of each word/pos-tag in the sentence.

    # END OF YOUR CODE
    retval = " ".join(word_list) if len(word_list) > 0 else None
    return retval


def get_pos_tags(pos_sent):
    retval = pos_sent.split(" ")
    retval = [re.search(r"(?<=[\/])\w+[$]{0,1}", word) for word in retval]
    retval = [word.group() for word in retval if (word != None)]
    return retval


def get_noun_phrases(pos_sent):
    """
    Find all simple noun phrases in pos_sent.

    A simple noun phrase is a single optional determiner followed by zero
    or more adjectives ending in one or more nouns.

    This function should return a list of noun phrases without tags.

    :param pos_sent: [string]
    :return: noun_phrases: [list]
    """
    noun_phrases = []

    # Your code goes here
    noun_phrases = re.findall(r"(?:\w+\/DT )?(?:\w+\/JJ )*?(?:\w+\/(?:NNP|NN|NNS) )+", pos_sent)
    noun_phrases = [get_words(word) for word in noun_phrases]
    # END OF YOUR CODE

    return noun_phrases


def read_stories(fname):
    stories = []
    with open(fname, 'r', encoding="utf8") as pos_file:
        story = []
        for line in pos_file:
            if line.strip():
                story.append(line)
            else:
                stories.append("".join(story))
                story = []
    return stories



def most_freq_noun_phrase(pos_sent_fname, verbose=True):
    """

    :param pos_sent_fname:
    :return:
    """
    story_phrases = {}
    story_id = 1
    for story in read_stories(pos_sent_fname):
        most_common = []
        # your code starts here
        most_common.extend(get_noun_phrases(story))
        # do stuff with the story
        dist = nltk.FreqDist([word.lower() for word in most_common])
        most_common = dist.most_common(3)
        # end your code
        if verbose:
            print("The most freq NP in document[" + str(story_id) + "]: " + str(most_common))
        story_phrases[story_id] = most_common
        story_id += 1

    return story_phrases

def most_freq_pos_tags(pos_sent_fname, verbose=True):
    """

    :param pos_sent_fname:
    :return:
    """
    story_tags = {}
    story_id = 1
    for story in read_stories(pos_sent_fname):
        most_common = []
        # your code starts here
        pos_tags = get_pos_tags(story)
        dist = nltk.FreqDist(pos_tags)
        most_common = dist.most_common(3)
        # do stuff with the story

        # end your code
        if verbose:
            print("The most freq pos tags in document[" + str(story_id) + "]: " + str(most_common))
        story_tags[story_id] = most_common
        story_id += 1

    return story_tags









# tests








def test_get_words():
    """
    Tests get_words().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_words() ...")
    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    retval = str(get_words(pos_sent))
    print("retval:", retval)

    gold = "All animals are equal but some animals are more equal than others"
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")


def test_get_pos_tags():
    """
    Tests get_pos_tags().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_pos_tags() ...")
    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    retval = str(get_pos_tags(pos_sent))
    print("retval:", retval)

    gold = str(['DT', 'NNS', 'VBP', 'JJ', 'CC', 'DT', 'NNS', 'VBP', 'RBR', 'JJ', 'IN', 'NNS'])
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")



def test_get_noun_phrases():
    """
    Tests get_noun_phrases().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_noun_phrases() ...")

    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print("input:", pos_sent)
    retval = str(get_noun_phrases(pos_sent))
    print("retval:", retval)

    gold = "['All animals', 'some animals', 'others']"
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")


def test_most_freq_noun_phrase(infile="fables-pos.txt"):
    """
    Tests get_noun_phrases().
    Do not modify this function.
    :return:
    """
    print("\nTesting most_freq_noun_phrase() ...")

    import os
    if os.path.exists(infile):
        noun_phrase = most_freq_noun_phrase(infile, False)
        gold1 = "[('the donkey', 6), ('the mule', 3), ('load', 2)]"
        gold2 = "[('the donkey', 6), ('the mule', 3), ('burden', 2)]"
        retval = str(noun_phrase[7])

        print("gold:\t", gold1)
        print("OR:\t", gold2)
        print("retval:\t", retval)

        assert retval == gold1 or retval == gold2, "test Fail:\n {} != {} OR {}".format(noun_phrase[7], gold1, gold2)
        print("Pass")
    else:
        print("Test fail: path does not exist;", infile)

def test_most_freq_pos_tags(infile="fables-pos.txt"):
    """
    Tests most_freq_pos_tags().
    Do not modify this function.
    :return:
    """
    print("\nTesting most_freq_pos_tags() ...")

    import os
    if os.path.exists(infile):
        pos_tags = most_freq_pos_tags(infile, False)
        gold = "[('DT', 28), ('NN', 24), ('IN', 21)]"
        retval = str(pos_tags[7])

        print("gold:\t", gold)
        print("retval:\t", retval)

        assert retval == gold, "test Fail:\n {} != {}".format(pos_tags[7], gold)
        print("Pass")
    else:
        print("Test fail: path does not exist;", infile)


def run_tests():
    test_get_words()
    test_get_pos_tags()
    test_get_noun_phrases()
    test_most_freq_noun_phrase()
    test_most_freq_pos_tags()


if __name__ == '__main__':

    # comment this out if you dont want to run the tests
    run_tests()

    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-i', dest="pos_sent_fname", default="blogs-pos.txt",  help='File name that contant the POS.')

    args = parser.parse_args()
    pos_sent_fname = args.pos_sent_fname

    most_freq_noun_phrase(pos_sent_fname)

