from nltk.corpus import wordnet as wn

def main():
	def process_words(w1, w2, filename):
		with open(filename, "w+") as f:
			print("Synsets for " + w1 + ": ", file=f)
			for synset in wn.synsets(w1):
				print("\t" + synset.name() + ", with definition: " + synset.definition(), file=f)

			print(file=f)

			first_synsets = wn.synsets(w1)[0]
			print("First synset of " + w1 + " is " + first_synsets.name() + ": ", file=f)
			print("\tHypernyms of " + first_synsets.name() + " are: " + str(first_synsets.hyponyms()), file=f)
			print("\tRoot Hypernyms of " + first_synsets.name() + " are: " + str(first_synsets.root_hypernyms()), file=f)

			print(file=f)

			paths = first_synsets.hypernym_paths()[0]
			print("Root Hypernym Path for " + w1 + ": ", file=f)
			[print("\t" + syn.name(), file=f) for syn in paths]

			print(file=f)

			print("The path similarity between " + first_synsets.name() + " and " + wn.synsets(w2)[0].name() + " (the first synset of " + w2 + ") is: \n\t" + str(first_synsets.path_similarity(wn.synsets(w2)[0])), file=f)

			print(file=f)
			print("The maximum similarity pairs in [\"dog.n.01\", \"man.n.01\", \"whale.n.01\", \"bark.n.01\", \"cat.n.01\"] are:", file=f)

			max_similarity = [0]
			word_pairs = [("", "")]
			assigned_list = ["dog.n.01", "man.n.01", "whale.n.01", "bark.n.01", "cat.n.01"]
			for i in range(len(assigned_list)):
				for j in range(i + 1, len(assigned_list)):
					if wn.synset(assigned_list[i]).path_similarity(wn.synset(assigned_list[j])) >= max(max_similarity):
						max_similarity.append(wn.synset(assigned_list[i]).path_similarity(wn.synset(assigned_list[j])))
						word_pairs.append((assigned_list[i], assigned_list[j]))
			result = {}
			for i in range(len(max_similarity)):
				result[word_pairs[i]] = max_similarity[i]
			max_val = max(max_similarity)
			for val in result.keys():
				if result[val] == max_val:
					print("\t" + str(val), file=f)

	process_words("car", "engine", "wordnet.txt")

if __name__ == "__main__":
	main()
