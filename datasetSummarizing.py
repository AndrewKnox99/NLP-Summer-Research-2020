import pandas as pd
import numpy as np


saveDirPre = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\preCovidArticlesIndexed.csv"
saveDirPost = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\postCovidArticlesIndexed.csv"
dfPre = pd.read_csv(saveDirPre, index_col=0, header=0)
dfPost = pd.read_csv(saveDirPost, index_col=0, header=0)

wfuCount = dfPre.loc[dfPre['Dataset_ID'] == 1].shape[0]
wfuCount += dfPost.loc[dfPost['Dataset_ID'] == 1].shape[0]

stanCount = dfPre.loc[dfPre['Dataset_ID'] == 0].shape[0]
stanCount += dfPost.loc[dfPost['Dataset_ID'] == 0].shape[0]

dadeCount = dfPre.loc[dfPre['Dataset_ID'] == 2].shape[0]
dadeCount += dfPost.loc[dfPost['Dataset_ID'] == 2].shape[0]

print("Number of articles from Wake Forest: " + str(wfuCount))
print("Number of articles from Stanford: " + str(stanCount))
print("Number of articles from Dade: " + str(dadeCount))

print("Dade:\n")
textListPre = dfPre.loc[dfPre['Dataset_ID'] == 2]
textListPre = textListPre.iloc[:, 2].values
textListPost = dfPost.loc[dfPost['Dataset_ID'] == 2]
textListPost = textListPost.iloc[:, 2].values


#Get avg number of sentences and words from text
def getSentenceCount(texts):
    docCount = 0
    sentenceCount = 0
    wordCount = 0
    for text in texts:
        docCount += 1
        if type(text) != str:
            continue
        for word in text:
            wordCount += 1
            if word[len(word) - 1] == '.':
                sentenceCount += 1
            elif word[len(word) - 1] == '!':
                sentenceCount += 1
            elif word[len(word) - 1] == '?':
                sentenceCount += 1
    avgSentences = sentenceCount / docCount
    avgWord = wordCount / docCount
    #print("Avg sentences: " + str(avgSentences))
    #print("Avg words: " + str(avgWord))
    return avgSentences, avgWord


sentence, word = getSentenceCount(textListPre)
sentence2, word2 = getSentenceCount(textListPost)
print("Avg sentences: " + str((sentence + sentence2)/2))
print("Avg words: " + str((word + word2)/2))