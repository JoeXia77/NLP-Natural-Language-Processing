
import nltk
import random
from collections import Counter
import math

# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings

corpus_path1 = './warpeace.txt'
corpus_path2 = './shakespeare.txt'

## return [ [word1,word2,word3...forming a sentence] , [word1,word2,word3...forming a sentence] ]
def load_corpus(corpus_path):
    f = open(corpus_path)
    content = f.read()
    
    paragraphs = content.split('\n\n')
    sentences = []
    for i in paragraphs:
        sentences.extend(nltk.tokenize.sent_tokenize(i))
    words = []

    for i in sentences:
        temp = nltk.tokenize.word_tokenize(i)

        words.append(temp)
    return words


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings

## input text
## output wordContext = [ (str1,(contextword1,contextword2)) , (str1,(context1,context2))] contextWord Number see n
def get_ngrams(n, text):
    
    text = ['<s>']*(n-1) + text + ['</s>']
    
    wordContext = []
    L = len(text)
    for i in range(n-1,L):
        wordContext.append((text[i],tuple(text[i-n+1:i])))
    return wordContext


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n, corpus_path):
    n_gram_lm = NGramLM(n)
    content = load_corpus(corpus_path)
    for sentence in content:
        n_gram_lm.update(sentence)
    return n_gram_lm


class NGramLM:
    def __init__(self, n: int):
        self.n = n
        
        ## { (str1,(context1,context2)): count , (str1,(context1,context2)): count }
        self.ngram_counts = Counter()
        
        ## { (contextword1,contextword2): count , (contextword1,contextword2): count }
        self.context_counts = Counter()
        
        self.vocabulary = set()
        
        self.randomWords = []
        
        self.randomWordsHasChange = 1
        
        

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    ## input: [str1,str2]
    def update(self, text):
        ## update ngram
        
        tempNGram = get_ngrams(self.n, text)

        tempGramCounter = Counter(tempNGram)

        for i,j in tempGramCounter.items():
            self.ngram_counts[i] += j
        
        ## update the next: self.context
        context = [x[1] for x in tempNGram]
        contextCounter = Counter(context)

        for i,j in contextCounter.items():
            self.context_counts[i] += j
            
        ## update v
        self.vocabulary = self.vocabulary | set(text)
        self.vocabulary.add('</s>')
        
        self.randomWordsHasChange = 1

        
    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    ## input a context and a word
    ## output the possi that this word will show up in this context

    def get_ngram_prob(self, word, context, delta= .0):
        ## possi = number of dict[(word,context)] / dict[context]
        if self.context_counts[context] == 0:
            return 1 / len(self.vocabulary)

        if delta == 0:
            possi = (self.ngram_counts[(word,context)]) / (self.context_counts[context])
        else:
            possi = (self.ngram_counts[(word,context)]+delta) / (self.context_counts[context]+delta*len(self.vocabulary))
        
        return possi
        
    
    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent, delta=.0):
        ori = get_ngrams(self.n, sent)
        total_possi = 0
        for combine in ori:            
            cur_possi = self.get_ngram_prob(combine[0],combine[1],delta)
            if cur_possi <= 0:
                cur_possi = float('-inf')
            else:
                cur_possi = math.log(cur_possi,2)
            total_possi+=cur_possi

        return total_possi
    
    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    ## input [[word1,word2],[word1,word2]]
    def get_perplexity(self, corpus):
        ## 
        totalPossi = 0
        totalWords = 0
        for sentence in corpus:
            totalWords += len(sentence)
            totalPossi += self.get_sent_log_prob(sentence)
        avePossi = totalPossi/totalWords
        perplexity = math.pow(2,(-avePossi))
        return perplexity
            
    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context, delta= .0) -> str:
        if self.randomWordsHasChange == 1:
            self.randomWords = list(self.vocabulary)
            self.randomWords.sort()
            self.randomWordsHasChange = 0
        
        boundry = random.random()
        totalPossi = 0
        
        for word in self.randomWords:
            possi = self.get_ngram_prob(word, context, delta)
            totalPossi+=possi
            if totalPossi>boundry:

                
                return word
        else:
            return 'the'
        '''
        
        ## use another alg
        ## del word that never appears
        
        ## build a rank list: [(possi,word),(possi,word)]
        possiRank = []
        for word in self.randomWords:
            logPossi = self.get_ngram_prob(word, context, delta)
            possi = 2 ** logPossi
            possiRank.append((possi,word))
        
        
        possiRank.sort(reverse = True)
        avePossi = sum([x[0] for x in possiRank])/len(possiRank)
        ## find the index whose possi == avePossi
        stopIndex = 0
        for i in range(len(possiRank)):
            if possiRank[i][0]<avePossi:
                stopIndex = max(i-1,0)
                break
        choosingRange = possiRank[:stopIndex+1]
        temp = sum([x[0] for x in choosingRange])
        boundry = random.random() * temp
        totalPossi = 0
        for i in choosingRange:
            totalPossi+=i[0]
            if totalPossi>boundry:
                ## print('possi',context,'-->',i[1],':',i[0])
                return i[1]
        else:
            return choosingRange[-1][1]
        '''
        
    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length, delta=.0):
        context = ['<s>'] * (self.n - 1)
        resultSentence = []
        count = 0
        superCount = 0
        while count<max_length and superCount<100:
            superCount+=1
            word = self.generate_random_word(tuple(context),delta)
            if word == '</s>':
                resultSentence.append(word)
                break
            ## could change to automatic identify
            if word == 'zu' or word == 'zwaggered':
                context = ['<s>'] * (self.n - 1)
                continue
            count+=1  
            resultSentence.append(word)
            context.append(word)
            del context[0]
        resultSentence = ' '.join(resultSentence)
        return resultSentence
          
    
for n in [1,3,5]:
    lm = create_ngram_lm(n,'./shakespeare.txt')
    print(n,'- gram')
    for i in range(5):
        print(lm.generate_random_text(10))
