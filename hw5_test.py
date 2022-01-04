import unittest, io
from hw5 import *
from compare_pandas import compare_frames
from contextlib import redirect_stdout

import string, pandas as pd, numpy as np, pickle as pkl
from sklearn.feature_extraction import stop_words
from nltk.stem import SnowballStemmer

'''
compare_pandas.py
aaou.txt, gotg1.txt, gotg2a.txt, gotg2b.txt, gw.txt, saguaro.txt, small1.txt, small2.txt, small3.txt
small1_cleaned.pkl, small2_cleaned.pkl, small3_cleaned.pkl, aaou_cleaned.pkl, gotg1_cleaned.pkl, saguaro_cleaned.pkl
small1_vector.pkl, small2_vector.pkl, small3_vector.pkl
aaou_vector.pkl, gotg1_vector.pkl, gotg2a_vector.pkl, gotg2b_vector.pkl, gw_vector.pkl, saguaro_vector.pkl
tfidf_small.pkl, tfidf.pkl, similarity_small.pkl, similarity.pkl, pretty_matrix_small.pkl, pretty_matrix.pkl
docfreqs_small.pkl, docfreqs.pkl
'''

class TestAssignment4(unittest.TestCase):
    
    def setUp(self):
        #self.maxDiff = None
        self.v0 = {}
        self.v1 = {'a' : 1}
        self.v2 = {'b' : 2}
        self.v3 = {'a' : 1, 'b' : 2, 'c' : 3}
        self.v4 = {'a' : 3, 'b' : 2, 'c' : 1}
        self.v5 = {'a' : 2, 'b' : 0, 'c' : 0.5, 'd' : 1.5}
        self.v6 = {'a' : 2, 'b' : 1, 'c' : 0, 'd' : 0.5}
        self.files_small = ['small1.txt', 'small2.txt', 'small3.txt']
        self.files_big = ['gotg2a.txt', 'gotg2b.txt', 'gotg1.txt', 'aaou.txt', 'gw.txt', 'saguaro.txt']
        self.vectors_small = [pkl.load(open('small' + str(i) + '_vector.pkl', 'rb')) for i in range(1, 4)]
        self.vectors_big = [pkl.load(open(fname[:-4] + '_vector.pkl', 'rb')) for fname in self.files_big]
        self.stemmer = SnowballStemmer('english')
        self.stops = {k for k in list(stop_words.ENGLISH_STOP_WORDS) + ['did', 'gone', 'ca']}
        
    def test_dot_product(self):
        self.assertEqual(0, dot_product(self.v0, self.v1))
        self.assertEqual(1, dot_product(self.v1, self.v1))
        self.assertEqual(1, dot_product(self.v1, self.v3))
        self.assertEqual(0, dot_product(self.v1, self.v2))
        self.assertEqual(4, dot_product(self.v2, self.v3))
        self.assertEqual(10, dot_product(self.v3, self.v4))
        self.assertEqual(6.5, dot_product(self.v4, self.v5))
        self.assertEqual(4.75, dot_product(self.v5, self.v6))
        
    def test_magnitude(self):
        self.assertEqual(0, magnitude(self.v0))
        self.assertEqual(1, magnitude(self.v1))
        self.assertEqual(2, magnitude(self.v2))
        self.assertEqual(np.sqrt(14), magnitude(self.v3))
        self.assertEqual(np.sqrt(14), magnitude(self.v4))
        self.assertEqual(np.sqrt(6.5), magnitude(self.v5))
        self.assertEqual(np.sqrt(5.25), magnitude(self.v6))
        
    def test_cosine_similarity(self):
        self.assertTrue(abs(1 - cosine_similarity(self.v1, self.v1)) < 0.0001)
        self.assertTrue(abs(1/np.sqrt(14) - cosine_similarity(self.v1, self.v3)) < 0.0001)
        self.assertTrue(abs(cosine_similarity(self.v1, self.v2)) < 0.0001)
        self.assertTrue(abs(2/np.sqrt(14) - cosine_similarity(self.v2, self.v3)) < 0.0001)
        self.assertTrue(abs(5/7 - cosine_similarity(self.v3, self.v4)) < 0.0001)
        self.assertTrue(abs(round(6.5/np.sqrt(91), 10) - round(cosine_similarity(self.v4, self.v5), 10)) < 0.0001)
        self.assertTrue(abs(round(4.75/np.sqrt(34.125), 10) - round(cosine_similarity(self.v5, self.v6), 10)) < 0.0001)
        
    def test_get_text(self):
        self.assertEqual(pkl.load(open('small1_cleaned.pkl', 'rb')), get_text('small1.txt'))
        self.assertEqual(pkl.load(open('small2_cleaned.pkl', 'rb')), get_text('small2.txt'))
        self.assertEqual(pkl.load(open('small3_cleaned.pkl', 'rb')), get_text('small3.txt'))
        self.assertEqual(pkl.load(open('aaou_cleaned.pkl', 'rb')), get_text('aaou.txt'))
        self.assertEqual(pkl.load(open('gotg1_cleaned.pkl', 'rb')), get_text('gotg1.txt'))
        self.assertEqual(pkl.load(open('saguaro_cleaned.pkl', 'rb')), get_text('saguaro.txt'))
        
    def test_vectorize(self):
        self.assertEqual(pkl.load(open('small1_vector.pkl', 'rb')), vectorize('small1.txt', self.stops, self.stemmer))
        self.assertEqual(pkl.load(open('small2_vector.pkl', 'rb')), vectorize('small2.txt', self.stops, self.stemmer))
        self.assertEqual(pkl.load(open('small3_vector.pkl', 'rb')), vectorize('small3.txt', self.stops, self.stemmer))
        self.assertEqual(pkl.load(open('gotg2a_vector.pkl', 'rb')), vectorize('gotg2a.txt', self.stops, self.stemmer))
        self.assertEqual(pkl.load(open('gotg2b_vector.pkl', 'rb')), vectorize('gotg2b.txt', self.stops, self.stemmer))
        self.assertEqual(pkl.load(open('gw_vector.pkl', 'rb')), vectorize('gw.txt', self.stops, self.stemmer))
        
    def test_get_doc_freqs(self):
        self.assertEqual(pkl.load(open('docfreqs_small.pkl', 'rb')), get_doc_freqs(self.vectors_small))
        self.assertEqual(pkl.load(open('docfreqs.pkl', 'rb')), get_doc_freqs(self.vectors_big))
        
    def test_tfidf(self):
        self.assertIsNone(tfidf(self.vectors_small), tfidf(self.vectors_big))
        self.assertEqual(pkl.load(open('tfidf_small.pkl', 'rb')), self.vectors_small)
        self.assertEqual(pkl.load(open('tfidf.pkl', 'rb')), self.vectors_big)
        
    def test_get_similarity_matrix(self):
        self.assertTrue(compare_frames(pd.read_pickle('similarity_small.pkl'), get_similarity_matrix(self.files_small, self.stops, self.stemmer)))
        self.assertTrue(compare_frames(pd.read_pickle('similarity.pkl'), get_similarity_matrix(self.files_big, self.stops, self.stemmer)))
        
    def test_matrix_pretty_string(self):
        self.assertEqual(pkl.load(open('pretty_matrix_small.pkl', 'rb')), matrix_pretty_string(pd.read_pickle('similarity_small.pkl')))
        self.assertEqual(pkl.load(open('pretty_matrix.pkl', 'rb')), matrix_pretty_string(pd.read_pickle('similarity.pkl')))
        
    def test_main(self):
        with io.StringIO() as buf, redirect_stdout(buf):
            main()
            self.assertEqual(pkl.load(open('pretty_matrix.pkl', 'rb')) + '\n', buf.getvalue())
        
    
test = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment4)
results = unittest.TextTestRunner().run(test)
print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 100) + ' / 100')
