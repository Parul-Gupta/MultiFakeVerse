import argparse, os
from tqdm.auto import tqdm
import random
import json
import pandas as pd
import glob

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

# Import external libraries: spaCy for tokenization, lemmatization and stopwords
import spacy
from spacy.lang.en import English                 # For other languages, refer to the SpaCy website: https://spacy.io/usage/models
from spacy.lang.en.stop_words import STOP_WORDS   # Also need to update stopwords for other languages (e.g. spacy.lang.uk.stop_words for Ukrainian)

# Import external libraries: gensim to create models and do some additional preprocessing
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Import external libraries: pyLDA for vis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

number_of_topics = 10
my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

def parse_args():
    parser = argparse.ArgumentParser("analyze the real-fake comparison outputs")
    parser.add_argument("--captions_folder", "-c", type=str, default="outputs/image_captions")
    args = parser.parse_args()
    return args

# Lemmatize tokens
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):   # Doing part of speech (PoS) tagging helps with lemmatization
    # Load the nlp pipeline, omitting the parser and ner steps of the workflow to conserve computer memory
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # For other languages, use models from step 2
    # nlp.max_length = 1500000 # Or other value, given sufficient RAM
    texts_out = []
    for text in tqdm(texts, desc="Lemmatizing all captions..."): # Run each of the documents through the nlp pipeline
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

# Preprocess texts
def gen_words(texts):
    final = [] # Create an empty list to hold tokens
    for text in tqdm(texts, desc="Processing lemmatized captions..."):
        new = simple_preprocess(text, deacc = True) 
        # If working with languages that employ accents, you can set deacc to False
        final.append(new)
    return (final)

# cleaning master function
def clean_caption(caption, bigrams=False):
    caption = caption.lower() # lower case
    caption = re.sub('['+my_punctuation + ']+', ' ', caption) # strip punctuation
    caption = re.sub('\s+', ' ', caption) #remove double spacing
    caption = re.sub('([0-9]+)', '', caption) # remove numbers
    caption_token_list = [word for word in caption.split(' ')
                            if word not in my_stopwords] # remove stopwords

    caption_token_list = [word_rooter(word) if '#' not in word else word
                        for word in caption_token_list] # apply word rooter
    if bigrams:
        caption_token_list = caption_token_list+[caption_token_list[i]+'_'+caption_token_list[i+1]
                                            for i in range(len(caption_token_list)-1)]
    caption = ' '.join(caption_token_list)
    return caption

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

def read_all_caption_files(args):
    files_list = glob.glob(f"{args.captions_folder}/**/*.json", recursive=True)
    captions_dict = {}
    for fl in tqdm(files_list, desc="Reading caption json files..."):
        dt = json.load(open(fl, "r"))
        for k,v in dt.items():
            captions_dict[f"{os.path.basename(fl).replace('.json', '')}_{k}"] = v
    
    return captions_dict

def topic_modelling_captions_nltk(args):
    captions_dict = read_all_caption_files(args)
    print(f"Analyzing {len(captions_dict)} captions...")
    captions_df = pd.DataFrame(list(captions_dict.items()), columns=["file_name", "caption"])
    del captions_dict
    captions_df['clean_caption'] = captions_df.caption.apply(clean_caption)

    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

    # apply transformation
    tf = vectorizer.fit_transform(captions_df['clean_caption']).toarray()

    # tf_feature_names tells us what word each column in the matric represents
    tf_feature_names = vectorizer.get_feature_names_out()

    # model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model = NMF(n_components=number_of_topics, random_state=0, l1_ratio=.5)
    model.fit(tf)
    no_top_words = 10
    print(display_topics(model, tf_feature_names, no_top_words))
    return

def topic_modelling_captions_spacy(args):
    captions_dict = read_all_caption_files(args)
    captions_list = list(captions_dict.values())
    lemmatized_texts = lemmatization(captions_list)
    print(lemmatized_texts[0][0:90]) # Print results to verify
    data_words = gen_words(lemmatized_texts) # Pass lemmatized_texts from previous step through the gen_words function
    bigram_phrases = gensim.models.Phrases(data_words, min_count=3, threshold=50)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=50)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    def make_bigrams(texts):
        return [bigram[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram[bigram[doc]] for doc in texts]
    
    data_bigrams = make_bigrams(data_words)
    data_bigrams_trigrams = make_trigrams(data_bigrams)

    # --Uncomment to print list of words showing bigrams and trigrams
    # print (data_bigrams_trigrams[0])

    id2word = corpora.Dictionary(data_bigrams_trigrams)

    # Represent dictionary words as tuples (index, frequency)
    corpus = []
    for text in tqdm(data_bigrams_trigrams, desc="converting to bag-of-words..."):
        new = id2word.doc2bow(text)
        corpus.append(new)
    
    # Retrieve individual words from tuples
    word = id2word[[0][:19][0]]   # Change the first number (currently 0) to see the various terms indexed in step 7.
    print(word)

    # Create LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=number_of_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,     
                                                # Change chunksize to increase or decrease the length of segments
                                                passes=50,         
                                                # Can do more passes but will increase the time it takes the block to run
                                                alpha="auto"
                                            )

    # Print topics
    lda_model.show_topics()

    # Output visualization
    vis_data = gensimvis.prepare(lda_model, corpus, id2word, R=15, mds='mmds')
    vis_data
    pyLDAvis.display(vis_data)
    pyLDAvis.save_html(vis_data, './topicVis' + str(number_of_topics) + '.html')

def bert_topic_modelling(args):
    from bertopic import BERTopic
    captions_dict = read_all_caption_files(args)
    captions_list = list(captions_dict.values())
    nlp = spacy.load('en_core_web_md', exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

    topic_model = BERTopic(embedding_model=nlp)
    topics, probs = topic_model.fit_transform(captions_list)

    fig = topic_model.visualize_topics()
    fig.show()


if __name__ == "__main__":
    args = parse_args()
    # topic_modelling_captions_nltk(args)
    topic_modelling_captions_spacy(args)
    # bert_topic_modelling(args)