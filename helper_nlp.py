from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class featureExtraction(object):
    """
    featureExtraction represents the stage in which the text data are processed,
    or reduced from words/strings to numbers
    
    Example usage:
        text = featureExtraction(train.ingredients)
        trainAsStrings = text.listToString()
        stemmedStrings = text.stem(trainAsStrings)
        vec = text.tfidfVectorize(stemmedStrings, max_features=2000, ngram_range=(1,1), \
                       lowercase=True, stop_words=None, max_df=0.5, min_df= 1)
        bow_train = text.bag_of_words(vec, stemmedStrings)

    """
    
    def __init__(self, data):
        """Assign raw data. Assumes dataframe """
        self.data = data
        
    def listToString(self):
        """Extract the unique ingredients. Assumes list of strings and 
        returns a string.
        E.g. input = ['romaine lettuce', 'black olives', 'salt', ...]
        E.g. output = 'romaine lettuce black olives salt ...'
        """ 
        words = [' '.join(item) for item in self.data]
        return words
    
    def stem(self, words):
        stemmedTokens = [stemmer.stem(w) for w in words] 
        return stemmedTokens
    
    def countVectorize(self, stemmedTokens, max_features=2500, ngram_range=(1,1), \
                  lowercase=True, stop_words=None, max_df=0.5, min_df= 1):
        """Tokenize and count words.
        1. Instantiate vectorizer 'vec'
        2. Fit: learn vocabulary and idf from training set"""
        
        vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range, \
                              lowercase=lowercase, stop_words=stop_words, max_df=max_df, min_df= min_df)
        vec.fit(stemmedTokens)
    
        return vec
    
    def tfidfVectorize(self, stemmedTokens, max_features=2500, ngram_range=(1,1), \
                  lowercase=True, stop_words=None, max_df=0.5, min_df= 1):
        """Tokenize, count, and weight the words.
        1. Instantiate vectorizer 'vec'
        2. Fit: learn vocabulary and idf from training set"""
        
        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, \
                              lowercase=lowercase, stop_words=stop_words, max_df=max_df, min_df= min_df)
        vec.fit(stemmedTokens)
        return vec
    
    def bag_of_words(self, vec, stemmedTokens):
        """Transform documents to document-term matrix"""
        bag_of_words = vec.transform(stemmedTokens).toarray()
        return bag_of_words