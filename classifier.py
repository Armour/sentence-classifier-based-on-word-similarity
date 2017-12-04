#!usr/bin/env python3

import nltk
from nltk.corpus import wordnet
from rake_nltk import Rake

def classifier(sentence, category):
    # download data
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    # init category
    category_list = []
    mapping = {}
    for k in category.keys():
        syn_k = wordnet.synsets(k, pos=wordnet.NOUN)[0]
        category_list.append(syn_k)
        mapping[syn_k.name()] = k

    # extract keywords
    r = Rake()
    r.extract_keywords_from_text(sentence)
    phrases = r.get_ranked_phrases_with_scores()

    # compute keywords score
    keywords = {}
    for score, phrase in phrases:
        tagged_tokens = nltk.pos_tag(nltk.word_tokenize(phrase))
        for token, tag in tagged_tokens:
            if tag.startswith('NN'):
                keywords[token] = keywords.get(token, 0) + score

    # computer similarity
    res = category
    for keyword in keywords:
        wordsets = wordnet.synsets(keyword, pos=wordnet.NOUN)
        if len(wordsets):
            for c in category_list:
                name = mapping[c.name()]
                res[name] = res.get(name, 0) + wordnet.path_similarity(c, wordsets[0])

    # return accumulative probability for this category
    return res

# Test
category = {
    # 'category_keyword': accumulative probability
    'sports': 0,
    'food': 0,
    'movie': 0,
    'technique': 0,
    'travel': 0
}
category = classifier('What sports do you like the best and why?', category)
print(category)
