# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:10:45 2021

@author: Sam
"""
# -*- coding: utf-8 -*-
"""
Python module for text sentiment analysis.
@authors: Sam Jasper and Lindy Bustabad
@date: October 16, 2021
"""
##########################################################################
#0. Import relevant libraries and load in data
##########################################################################
import numpy as np
import pandas as pd
from datetime import datetime
import math
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
import string
import gensim
import copy
nltk.download( 'vader_lexicon' )
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_module import sentiment
from nrclex import NRCLex


################################################################################
################################################################################
################################################################################


def preprocess(csv):
    """
    Read-in and pre-processing of Reddit CSV file. 
    :param csv: csv file to read-in
    :return: df_reddit data frame of processed data; df_reddit_highimp data frame of high impact posts
    """
    df_reddit = pd.read_csv (csv)

    #Convert the timestamp column from string to datetime
    df_reddit['timestamp'] = pd.to_datetime(df_reddit['timestamp'])

    #Create a new column (all_text) that includes the body and title columns (if not a comment)
    df_reddit['all_text'] = None

    #Copy over the title twice (i.e., double weight words in the title)
    #Cap impact values at 50 (to remove outliers)
    for index, row in df_reddit.iterrows():
        if row['score'] > 50:
            df_reddit['score'][index] = 50
        if row['body'] != row['body']: #Titles with no body. Skip nans in the body column (nans are not equal to themselves)
            df_reddit['all_text'][index] = row['title'] + ' ' + row['title']
        elif row['title'] != 'Comment': #Titles with body.
            df_reddit['all_text'][index] = row['title'] + ' ' + row['title'] + ' ' + row['body'] 
        else: #Comments
            df_reddit['all_text'][index] = row['body'] #Comments do not have titles
    
    #Create a new data frame that only includes high impact posts (top 20%)
    cutoff = np.quantile(df_reddit['score'],0.8)
    df_reddit_highimp = df_reddit[df_reddit['score'] >= cutoff]

    return df_reddit, df_reddit_highimp

################################################################################

def create_term_vec(df_reddit):
    """
    Create term vector from data frame.
    :param df_reddit: data frame
    :return: term_vec term vector
    """
    # Remove punctuation, then tokenize documents
    punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
    term_vec = [ ]

    # Create term vector
    for d in df_reddit['all_text']:
        if d != d: #Skip NaNs in the body column (NaNs are not equal to themselves)
            continue
        else:
            d = d.lower()
            d = punc.sub( '', d )
            term_vec.append( nltk.word_tokenize( d ) )

    return term_vec

################################################################################

def remove_stopwords(term_vec):
    """
    Remove stop words from term vector.
    :param term_vec: term vector
    :return: term_vec term vector without stop words
    """
    stop_words = nltk.corpus.stopwords.words( 'english' )

    for i in range( 0, len( term_vec ) ):
        term_list = [ ]

        for term in term_vec[ i ]:
            if term not in stop_words:
                term_list.append( term )

        term_vec[ i ] = term_list
    
    return term_vec

################################################################################

def porter_stem(term_vec):
    """
    Performs porter stemming on term vector. 
    :param term_vec: term vector without stop words
    :return: term_vec term vector without stop words and porter stemmed
    """
    porter = nltk.stem.porter.PorterStemmer()

    for i in range( 0, len( term_vec ) ):
        for j in range( 0, len( term_vec[ i ] ) ):
            term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

    return term_vec

################################################################################

def count_unique(term_vec):
    """
    Prints transposed term-frequency list for easier viewing
    :param term_vec: term vector
    :return: df_uniqcount data frame of unique terms
    """
    #Get unique terms in term vector
    term_list = set(term_vec[0])
    for i in range(1,len(term_vec)):
        term_list = term_list.union(term_vec[i])
        
    term_list = sorted(term_list)

    # Count occurrences of unique terms in each document of term vector
    n = len( term_list )
    freq = [ ]
    for i in range( 0, len( term_vec ) ):
        freq.append( [ 0 ] * n )
        for term in term_vec[ i ]:
            pos = term_list.index( term )
            freq[ -1 ][ pos ] += 1

    #Prints transposed term-frequency list for easier viewing
    # for i in range( 0, len( term_list ) ):
    #     print( f'{term_list[ i ]: <{20}}', end='' )
    #     for j in range( 0, 4 ): 
    #         print( f'{freq[ j ][ i ]:4d} ', end='' )
    #     print( '' )

    #Store count of unique terms (across all posts) in a data frame
    df_uniqcount = pd.DataFrame(columns = ['Term','Frequency'])
    for i in range( 0, len( term_list ) ):
        # print( f'{term_list[ i ]: <{20}}', end='' )
        count = 0
        for j in range( 0, len( term_vec ) ): #Loop through term vector and add each occurrence of every distinct word
            count = count + freq[j][i]
        #print( count )
        df_uniqcount = df_uniqcount.append({'Term':term_list[i],'Frequency':count}, ignore_index=True)

    return df_uniqcount

################################################################################

def impact_score(term_vec_us):
    """
    Creates data frame of impact scores.
    :param term_vec_us: term vector of unstemmed terms
    :return: df_impact_score data frame of impact scores
    """
    #Create object for sentiment analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    #Create dataframe to store results
    df_impact_score = pd.DataFrame(columns =  ["Impact_Score", "Sentiment_Score"])
    for i in range(0, len(term_vec_us)):
        impact_score = df_reddit['score'][i]
        sentence = " ".join(term_vec_us[i])
        sentiment_score = sentiment_analyzer.polarity_scores( sentence )
        df_impact_score.loc[i] = [impact_score,sentiment_score['compound']]

    return df_impact_score

################################################################################

def nrc_sentiment(term_vec_us_highimp):
    """
    Uses NRCLexicon package to associate sentiments with terms in term vector.
    :param term_vec_us_highimp: term vector of unstemmed terms for high impact posts
    :return: sentiment_count data frame of sentiments and counts
    """
    #Create empty data frame to append
    sentiment_count = pd.DataFrame(columns = ['count'], index = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy'])
    sentiment_count.loc['fear'] = [0]
    sentiment_count.loc['anger'] = [0]
    sentiment_count.loc['anticipation'] = [0]
    sentiment_count.loc['trust'] = [0]
    sentiment_count.loc['surprise'] = [0]
    sentiment_count.loc['positive'] = [0]
    sentiment_count.loc['negative'] = [0]
    sentiment_count.loc['sadness'] = [0]
    sentiment_count.loc['disgust'] = [0]
    sentiment_count.loc['joy'] = [0]

    # Loop through unstemmed term vector for high impact posts. (stop words have been removed)
    # For each term in the sentiment dictionary, add one value to that sentiment
    for row in term_vec_us_highimp:
        for word in row:
            sentiment = NRCLex(word)
            #If no emotions are associated with a word, move to the next word (to improve efficiency)
            if not sentiment.affect_list:
                continue
            for affect in sentiment.affect_list:
                sentiment_count.loc[affect] += 1/sum(sentiment.raw_emotion_scores.values())

    sentiment_count.to_csv('sentiment_count.csv',index = True)
    return sentiment_count


################################################################################

def valence_arousal(term_vec_us_highimp):
    """
    Uses NRCLexicon package to associate sentiments with terms in term vector.
    :param term_vec_us_highimp: term vector of unstemmed terms for high impact posts
    :return: sentiment_count data frame of sentiments and counts
    """
    #Create empty data frame for impact score, valence, arousal, and corresponding emotion
    df_impact_valaro = pd.DataFrame(columns =  ["Impact_Score", "Valence", "Arousal", "Emotion"])
    
    #Loop through high impact rows 
    i = 0
    for index, row in df_reddit_highimp.iterrows():
        impact_score = df_reddit_highimp['score'][index]
        sentence = term_vec_us_highimp[i]
        valence = sentiment.sentiment( sentence )['valence']
        arousal = sentiment.sentiment( sentence )['arousal']
        emotion = sentiment.describe(valence, arousal)
        
        df_impact_valaro.loc[i] = [impact_score, valence, arousal, emotion]
        i += 1
    
    #Write data frame to csv (exclude data points without valence)
    df_impact_valaro[df_impact_valaro['Valence'] != 0].to_csv('df_impact_valaro.csv',index = False)
    
    return df_impact_valaro

################################################################################
################################################################################
################################################################################


##########################################################################
#1. Data Read-in and Pre-processing
########################################################################## 
df_reddit, df_reddit_highimp = preprocess('reddit_vm_combined.csv')

##########################################################################
#2. Exploratory Data Analysis
##########################################################################
#Look at maximum and minimum values
min(df_reddit['timestamp'])
max(df_reddit['timestamp'])

df_reddit[df_reddit['title'] != 'Comment'].count() #464 posts (includes title. Not all posts have bodyies)
df_reddit[df_reddit['title'] == 'Comment'].count() #1116 comments (no title or URL)

##########################################################################
#3. Convert text into term vectors, remove stop words, and conduct Porter stemming
##########################################################################
#Create term vector
term_vec = create_term_vec(df_reddit)

#Remove stopwords
term_vec = remove_stopwords(term_vec)

#Store unstemmed term vector for sentiment analysis
term_vec_us = copy.deepcopy(term_vec)
    
# Porter stem remaining terms
term_vec = porter_stem(term_vec)

# ##########################################################################
# #4. Look at count of unique terms
# ##########################################################################

df_uniqcount = count_unique(term_vec)

#Look at most common words across all documents
df_uniqcount['Frequency'] = pd.to_numeric(df_uniqcount['Frequency']) #Convert Frequency column to integer for next line
df_uniqcount.nlargest(10, columns="Frequency").tail(10)

##########################################################################
#5. Sentiment Analysis
##########################################################################
#a. Post impact vs sentiment compound score
df_impact_score = impact_score(term_vec_us)

#Write impact scores dataframe to a CSV
df_impact_score.to_csv('Sentiment_Impact_Score.csv',index = False)

#b. Subset unstemmed term vector to only include high impact posts
cutoff = np.quantile(df_reddit['score'],0.8)
df_reddit_highimp = df_reddit[df_reddit['score'] >= cutoff]
term_vec_us_highimp = [term_vec_us[i] for i in df_reddit_highimp.index]

#NRCLexicon sentiment frequency count
sentiment_count = nrc_sentiment(term_vec_us_highimp)
print(sentiment_count)

#c. Valence and arousal scores for posts/comments
df_impact_valaro = valence_arousal(term_vec_us_highimp)

#Print out sentiments associated with valence and arousal for high-impact posts
print(df_impact_valaro.groupby("Emotion").size().sort_values(ascending=False))
