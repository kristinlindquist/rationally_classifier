from collections import Counter
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
import os
import umap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud


def get_top_topic_words(token_lists, labels, k=None):
    if k is None:
        k = len(np.unique(labels))
    topics = ['' for _ in range(k)]
    for i, c in enumerate(token_lists):
        topics[labels[i]] += (' ' + ' '.join(c))
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
    topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

    return topics

def get_coherence(model, token_lists, measure='c_v'):
    cm = CoherenceModel(
        topics=get_top_topic_words(token_lists, model.cluster_model.labels_),
        texts=token_lists,
        corpus=model.corpus,
        dictionary=model.dictionary,
        coherence=measure
    )
    return cm.get_coherence()

def get_silhouette(model):
    return silhouette_score(
        model.vec[model.method],
        model.cluster_model.labels_
    )

def plot_proj(embedding, labels):
    n = len(embedding)
    counter = Counter(labels)
    for i in range(len(np.unique(labels))):
        plt.plot(
            embedding[:, 0][labels == i],
            embedding[:, 1][labels == i],
            '.',
            alpha=0.5,
            label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100)
        )
    plt.legend()


def visualize(model):
    reducer = umap.UMAP()
    print('Performing UMAP projection ...')
    vec_umap = reducer.fit_transform(model.vec[model.method])
    print('Done performing UMAP projection.')
    plot_proj(vec_umap, model.cluster_model.labels_)
    dr = './images/{}'.format(model.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/2D_vis')

def get_wordcloud(model, token_lists, topic, labels):
    print('Generating wordcloud for topic {} ...'.format(topic))
    # labels = model.cluster_model.labels_
    tokens = [' '.join(_) for _ in np.array(token_lists)[labels == topic]]

    vectorizer = CountVectorizer(ngram_range=(1, 3))
    sum_words = vectorizer.fit_transform(tokens).sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        collocations=False,
        min_font_size=8
    ).fit_words(dict(words_freq))

    plt.figure(figsize=(10, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    dr = './images/{}'.format(model.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/Topic' + str(topic) + '_wordcloud')
    print('Wordcloud created for topic {}.'.format(topic))
