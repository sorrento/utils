import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_preprocessing(doc_list, kwargs):
    """Tf-IDF vectorization of corpus

    Parameters
    ----------
    doc_list : iterable
        Iterable list of texts
    kwargs : dict
        Dictionary with hyperparameters

    Returns
    -------
    vector_matrix, vocab, doc_freq:
        vector_matrix : NumPy array
            TF-IDF value matrix
        vocab : dict
            Token vocabulary dictionary
        doc_freq : NumPy array
            Document frequency for each token in vocab
    """

    if kwargs.get("tfidf_stop_words"):
        if kwargs.get("tfidf_strip_accents"):
            accents = 'ascii'
            with open('./textmining/classification/data/stopwords_ascii.txt', 'r') as fs:
                stopwords = fs.read().splitlines()
        else:
            accents = None
            with open('./textmining/classification/data/stopwords.txt', 'r') as fs:
                stopwords = fs.read().splitlines()
    else:
        stopwords = None
        accents = 'ascii' if kwargs.get("tfidf_strip_accents") else None

    text_vectorizer = TfidfVectorizer(
        strip_accents=accents,
        analyzer=kwargs.get("tfidf_analyzer"),
        stop_words=stopwords,
        ngram_range=(kwargs.get("tfidf_ngram_range_min"),
                     kwargs.get("tfidf_ngram_range_max")),
        max_df=kwargs.get("tfidf_max_df"),
        min_df=kwargs.get("tfidf_min_df"),
        norm=None,
    )

    vector_matrix = text_vectorizer.fit_transform(doc_list)
    # Obtain vocabulary and document frequency
    vocab = np.array(text_vectorizer.get_feature_names())
    doc_freq = 1.0 / text_vectorizer.idf_

    return vector_matrix, vocab, doc_freq