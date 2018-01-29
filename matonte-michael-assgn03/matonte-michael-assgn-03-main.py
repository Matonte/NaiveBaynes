import statistics
from collections import defaultdict
from itertools import product
import numpy as np
from math import log
#  Remove words common to both training reviews 

def getPrior(pos, neg):
    pos_occurences = len(pos)
    neg_occurences = len(neg)
    total_occ = pos_occurences + neg_occurences
    pos_prior = pos_occurences / total_occ
    neg_prior = neg_occurences / total_occ
    return pos_prior, neg_prior, pos_occurences, neg_occurences, total_occ

def getProbability(sentence, sentiment, prior, class_size):
    if sentiment == "+":
        reviews = prReviews
    if sentiment == "-":
        reviews = nrReviews    
    
    score = prior
    length_score=0
    sentence_words_set = set(sentence.split())   
    sentence_words_in_reviews = sentence_words_set.intersection(reviews)
    count = len(sentence_words_in_reviews)
   
    # conditional probability of a word given a certain sentiment
    if count > 0:
        prob_word_sentiment = count / class_size
        score = score * prob_word_sentiment        
    return score 

def getWordCount(sentence, hidden_layer):
    length_score=0
    sentence_words_set = set(sentence.split())
    length_score = abs(( statistics.mean(populateRawHiddenLayer(tReviews[i],hidden_layer)) - len(sentence_words_set)) / statistics.stdev(populateRawHiddenLayer(tReviews[i],hidden_layer)))
    return length_score

def populateRawHiddenLayer(sentence,hidden_layer):
    sentence_words_set = set(sentence.split())
    hidden_layer.append(len(sentence_words_set))
    return hidden_layer

# ------- main -----------
if __name__ == '__main__':
    
    with open("hotelNegT-train.txt","r") as fn:
        negativeReviews = fn.read()
    with open("hotelPosT-train.txt","r") as fp:
        positiveReviews = fp.read()

    nr = negativeReviews.split("\n")
    pr = positiveReviews.split("\n")
    nrIds = []
    prIds = [] 
    hidden_layer = []

    nrReviews = set()
    prReviews = set()
    
    for line in nr :
        w = line.split("\t")
        if w != [""]: 
            nrIds.append(w[0])
            # nrReviews.append(w[1])
            set_of_words = set(w[1].split())
            nrReviews.update(set_of_words)  # split string into words

    print('  len nr', len(nrReviews))

    for line in pr :
        w = line.split("\t")
        if w != [""]: 
            prIds.append(w[0])
            #prReviews.append(w[1])
            set_of_words = set(w[1].split())
            prReviews.update(set_of_words)  # split string into words            
            
    #print(prReviews)
    print('  len pr', len(prReviews))    
    
    #--- discard words common to both sets from each set ---
    common = nrReviews.intersection(prReviews)
    nrReviews.difference_update(common)
    prReviews.difference_update(common)
    print(' lengths w/o common:  Neg', len(nrReviews), ' Pos', len(prReviews))


    with open("pretest.txt","r") as t:
        test = t.read()

    tr = test.split("\n")

    tIds = []
    tReviews = []    

    fout = open('matonte-michael-assgn3-test-output.txt','w')

    for line in tr :
        w = line.split("\t")
        if w != [""]: 
            tIds.append(w[0])
            tReviews.append(w[1])

    pos_prior, neg_prior, pos_o, neg_o, total_o = getPrior(prReviews, nrReviews)
    neg_count = 0
    pos_count = 0
    true_count = 0 
    false_count = 0
    tf_deficit = 6
    score_factor = 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    for i in range (len(tReviews)):
        statistics.mean(populateRawHiddenLayer(tReviews[i],hidden_layer)) 
    while abs(tf_deficit) > 5: 
         true_count= 0
         false_count = 0
         for i in range (len(tReviews)):
           sentRatio = 1
           PosProb = getProbability(tReviews[i],"+", pos_prior, pos_o) 
           NegProb = getProbability(tReviews[i],"-", neg_prior, neg_o)
           WordCount = .1 * getWordCount(tReviews[i],hidden_layer)
           if PosProb > NegProb :
                sentRatio = 1.9* PosProb / NegProb
           if PosProb < NegProb :
                sentRato = 1.9 * NegProb / PosProb
           
           FTScore= sentRatio
           if  FTScore < score_factor:
                true_count += 1
           else:
                false_count += 1
         tf_deficit = true_count - false_count
         if tf_deficit > 5:
             score_factor -= .01
         if tf_deficit < -5:
             score_factor += .01
         print (tf_deficit)
         print (score_factor)
    
    count = 0
    correct = 0
    true_count= 0
    false_count = 0
    for i in range (len(tReviews)):
        sentRatio=1
        PosProb = getProbability(tReviews[i],"+", pos_prior, pos_o) 
        NegProb = getProbability(tReviews[i],"-", neg_prior, neg_o)
        if PosProb > NegProb :
            sentRatio = 1.9 *  PosProb / NegProb
        if PosProb < NegProb :
            sentRato = 1.9 * NegProb / PosProb
        WordCount = .1 * getWordCount(tReviews[i],hidden_layer)
        FTScore=  sentRatio
        print(PosProb,NegProb,FTScore)
        if PosProb < NegProb and FTScore < score_factor:
            fout.write ( tIds[i] + "    NEG   TRUE \n"   )
            neg_count += 1
            true_count += 1
            count += 1 
            if count < 215 :
                correct +=1
                
        elif PosProb < NegProb and FTScore >=  score_factor:
            fout.write ( tIds[i] + "    NEG   FALSE \n"   )
            neg_count += 1
            false_count += 1
            count += 1 
            if count >= 215 :
                correct +=1
                
        elif PosProb >= NegProb and FTScore >=  score_factor:
            fout.write ( tIds[i] + "    POS  FALSE \n"   )
            pos_count += 1
            false_count += 1
            count += 1 
            if count >= 215 :
                correct +=1
                
        else:
            fout.write ( tIds[i] + "    POS  TRUE \n"   )
            pos_count += 1
            true_count += 1
            count += 1 
            if count < 215 :
                correct +=1
                
    print (correct/count*100, " % right")
    print('COUNTS  Pos:', pos_count, ' Neg:', neg_count, 'True:', true_count, 'False:', false_count)


