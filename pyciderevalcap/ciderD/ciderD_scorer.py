#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import defaultdict
from collections import Counter
import numpy as np
import pdb
import math
import six
from six.moves import cPickle
import os

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

stopwords_ = None

def init_stopwords():
    global stopwords_
    stopwords_ = stopwords_ or list(stopwords.words('portuguese'))
    

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n, sigma=self.sigma, alpha=self.alpha, penalize_repetition=self.penalize_repetition)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def copy_empty(self):
        new = CiderScorer(df_mode="corpus", n=self.n, sigma=self.sigma, alpha=self.alpha, penalize_repetition=self.penalize_repetition)
        new.df_mode = self.df_mode
        new.ref_len = self.ref_len
        new.document_frequency = self.document_frequency
        return new

    def __init__(self, df_mode="corpus", test=None, refs=None, n=4, sigma=6.0, alpha=None, penalize_repetition=False):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.test = []
        self.refs = []
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        self.ref_len = None
        if self.df_mode != "corpus":
            pkl_file = cPickle.load(open(os.path.join('data', df_mode + '.p'),'rb'), **(dict(encoding='latin1') if six.PY3 else {}))
            self.ref_len = np.log(float(pkl_file['ref_len']))
            self.document_frequency = pkl_file['document_frequency']
        self.cook_append(test, refs)
        self.alpha = alpha
        self.penalize_repetition = penalize_repetition
        init_stopwords()


    def clear(self):
        self.crefs = []
        self.ctest = []
        self.refs = []
        self.test = []

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.refs.append(refs)
            self.test.append(test)
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def compute_penalty_by_repetition(hyp, ref):
            # tokenize only words
            tokenizer = RegexpTokenizer(r'[A-Za-z]+')
            tokens_hyp = tokenizer.tokenize(hyp)
            # clean_hyp = [token for token in tokens_hyp if token not in stopwords_]

            tokens_ref = tokenizer.tokenize(ref[0])
            # clean_ref = [token for token in tokens_ref if token not in stopwords_]

            word_freq_hyp = Counter(tokens_hyp)
            word_freq_ref = Counter(tokens_ref)

            sum = 0
            counter = 0
            for word, freq in word_freq_hyp.items():
                # words in the hypothesis but that are not in the reference
                if word_freq_ref.get(word, None) is not None:
                    diff = abs(word_freq_ref[word] - freq)
                    sum += np.exp(1 / (1 + diff))
                    counter += 1

            return 1 if counter == 0 else sum / (np.e * counter)

            # no penalty
            return 1

        def sim(sent_hyp, vec_hyp, sent_ref, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref, alpha=None,
                    penalize_repetition=False):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))

                
                if alpha is None:
                    # vrama91: added a length based gaussian penalty
                    val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
                else:
                    #val[n] *= np.e ** (-abs(delta)/float(length_ref))
                    val[n] *= np.e ** (-(delta**2)/(float(alpha)*float(length_ref)**2))

                if n == 0 and penalize_repetition:
                    penalty = compute_penalty_by_repetition(sent_hyp, sent_ref)
                    val[n] = val[n]*penalty

            return val

        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))
        #elif self.df_mode == "coco-val-df":
            # if coco option selected, use length of coco-val set
        #    self.ref_len = np.log(float(40504))

        scores = []
        for i, (test, refs) in enumerate(zip(self.ctest, self.crefs)):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(self.test[i], vec, self.refs[i], vec_ref, norm, norm_ref, length, length_ref,
                             alpha=self.alpha, penalize_repetition=self.penalize_repetition)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        if self.df_mode == "corpus":
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            # assert to check document frequency
            assert(len(self.ctest) >= max(self.document_frequency.values()))
            # import json for now and write the corresponding files
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)

