###IMPORTS###############
import cPickle
import operator

from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


##########################

class Stem_word(object):

    def __init__(self,k,v,model):
        self.parent = k
        self.childs = v
        l = [k] + v
        l = sorted(l,key=lambda x:len(x))
        self.stem = l[0]
        self.cosine_dist = list(map(lambda x: (x, cosine_similarity([model[k]], [model[x]])[0][0]), v))
        self.stem_dict = {}




class Word2Vec_stemmer(Stem_word):
    def __init__(self, model_file = 'nepal_model_for_stem', global_model_file = 'wiki.bin',
    alpha = 0.9, beta = 0.7, prefix = 2, m = 3, lambda_val = 0.9):
 
        self.model = Word2Vec.load(model_file)
        self.global_model = KeyedVectors.load_word2vec_format(global_model_file)
        self.beta = beta
        self.alpha = alpha        
        self.prefix = prefix
        self.m = m
        self.total_list = self.model.wv.vocab.keys()
        self.__clean_word_list()
        self.word2candidate_stems = {}
        self.total_list = set(self.total_list)
        self.__generate_candidate_stems()         
        candidate_words = self.total_list.difference(set(self.word2candidate_stems.keys()))
        print "candidates generated"
        # with open('non_candidate_words.txt','r') as fid:
        #     fid.write(str(candidate_words))
        self.Stemmed_Words = []       
        self.__word_to_list = {}
        self.stem_dict = {}
        self.union_dict = {}
        self.lambda_val = lambda_val * 0.1
        # self.b = (a_val//2) * 0.1
        self.gamma = self.__find_gamma()
        print "gamma calculated"
        #print self.gamma
        for i in self.model.wv.vocab.keys():
            self.union_dict[i] = i 
        self.__refining_and_stem_identification()
        

    def __clean_word_list(self):
        self.total_list[:] = [x for x in self.total_list if len(x) > 0]
        self.total_list[:] = [x for x in self.total_list if (ord(x[0])>= 97 and ord(x[0])<=122)]
        self.total_list[:] = list(set(self.total_list))
         
    def __find_gamma(self):
        word_to_gamma = {}
        for w1, c_list in self.word2candidate_stems.iteritems():
            max_gamma = 0.0     
            for w2 in c_list:
                if w1 == w2:
                    continue
                cos_sim = cosine_similarity([self.model[w1]], [self.model[w2]])[0][0]
                if cos_sim > max_gamma:
                    max_gamma = cos_sim
            
            word_to_gamma[w1] = self.lambda_val * max_gamma
        
        return word_to_gamma
                

    def __find_lcs(self, word1, word2):
        n = len(word1)
        m = len(word2)
        lcs = [[0]*(m+1) for i in xrange(n+1)]
        for i in xrange(1,n+1):
            for j in xrange(1,m+1):
                if(word1[i-1] == word2[j-1]):
                    lcs[i][j] = 1+lcs[i-1][j-1]
                else:
                    lcs[i][j] = max(lcs[i][j-1], lcs[i-1][j])
        
        return lcs[n][m]

    def __find_match(self, longer_word, shorter_word):
        if len(shorter_word) < self.m:
            return False
        return ((longer_word[:self.prefix]==shorter_word[:self.prefix]) and \
        (len(shorter_word[self.prefix:]) >= 2 and len(longer_word[self.prefix:]) >= 2) and \
    (float(self.__find_lcs(shorter_word[self.prefix:], longer_word[self.prefix:])))) > (self.alpha*len(shorter_word[self.prefix:]))




    def __generate_candidate_stems(self):
        taken_words = []

        for word in self.total_list:

            candidates = []
            if len(word) < self.m:
                continue
            n1 = len(word)
            for cds in self.total_list:
                w1 = None
                w2 = None
                if len(cds) > n1:
                    w1 = cds
                    w2 = word
                else:
                    w1 = word
                    w2 = cds
                if self.__find_match(w1, w2) == True:
                    candidates.append(cds)

            try:
                candidates.remove(word)
            except:
                pass        
            self.word2candidate_stems[word] = candidates

    
    def __refining_and_stem_identification(self):
        word2val = {}
        for (k,v) in self.word2candidate_stems.iteritems():
            lenK = len(k[self.prefix:])
            for cds in v:
                val = cosine_similarity([self.model[k]], [self.model[cds]])[0][0]
                if  val> self.gamma[k]:
                    if cds in word2val.keys():
                        pval = word2val[cds][0]
                        if pval < val:
                            word2val[cds] = (val,k)
                    else:
                        word2val[cds] = (val, k)
    
        print "word2val done"
        for (k,v) in word2val.iteritems():
            self.union(v[1],k)                
        print "union done"
        taken = []
        
        for i in self.model.wv.vocab.keys():
            p = self.find_parent(i)
            try:
                self.stem_dict[p].append(i)
            except:
                self.stem_dict[p] = [i]
        print "stem dict calculated"
        
        new_stem_dict = {}
        for k, v in self.stem_dict.iteritems():
            wordlist = [k] + v
            maxSim = self.__find_max_sim(wordlist)
            if maxSim <= 0.0:
                new_stem_dict[k] = v
                continue
             
            new_clus_list = self.__refine_cluster(maxSim, wordlist)
            new_stem_dict[k] = new_clus_list[1:]
        
            
        self.stem_dict = {}
        self.stem_dict = new_stem_dict.copy()    
        for ind, (k, v) in enumerate(new_stem_dict.iteritems()):
            self.Stemmed_Words.append(Stem_word(k,v,self.model))
            self.__word_to_list[k] = ind
            for i in v:
                self.__word_to_list[i] = ind
        print "word to list done"
    
    
    def __find_max_sim(self, word_list):
        maxSim = 0.0
        n = len(word_list)
        sim = 0.
        for i in range(n-1):
            for j in range(i+1, n):
                w1, w2 = word_list[i], word_list[j]
                if w1 == w2:
                    continue
                try:
                    sim = cosine_similarity([self.global_model[w1]], [self.global_model[w2]])[0][0]
                except:
                    maxSim = -1.0
                    break
                
                maxSim = max(maxSim, sim)
            
            if maxSim < 0.0:
                break
        return maxSim
    
    
    def __refine_cluster(self, maxSim, word_list):
        key, word_list = word_list[0], word_list[1:]
        new_clus_list = [key]
        old_len = 0
        new_len = len(new_clus_list)
        all_flag = 0
        while old_len < new_len:
            old_len = new_len
            
            
            for w in new_clus_list:
                w_sim  = self.global_model.most_similar(positive=[w], topn=1)

                for w1 in word_list:
                    if w1 in new_clus_list:
                        continue
                
                
                    w1_sim = self.global_model.most_similar(positive=[w1], topn=1)
                
                    if (w1_sim == w) or (w_sim == w1):
                        ww1_sim = cosine_similarity([self.global_model[w]], [self.global_model[w1]])[0][0]
                        if ww1_sim > self.beta * maxSim:
                            new_clus_list.append(w1)
                            new_len = len(new_clus_list)
        
        return new_clus_list
            
            
            
 
                

    def union(self,word1, word2):
        parent1 = self.find_parent(word1)
        parent2 = self.find_parent(word2)
        if len(parent1) > len(parent2):
            self.union_dict[parent1] = parent2
        else:
            self.union_dict[parent2] = parent1

    def find_parent(self,word):
        while self.union_dict[word] != word:
            word = self.union_dict[word]
        
        return word

    def word_info(self,word):
        try:
            ind = self.__word_to_list[word]
        except:
            print "word not in corpus"
            return
        Stem_info = self.Stemmed_Words[ind]
        print " word stem" ,Stem_info.stem
        print "word parent", Stem_info.parent
        print "cosine distances", Stem_info.cosine_dist
 


    def get_stem(self,word):
        try:
            ind = self.__word_to_list[word]
        except:            
            return word
        Stem_info = self.Stemmed_Words[ind]
        return Stem_info.stem
        
