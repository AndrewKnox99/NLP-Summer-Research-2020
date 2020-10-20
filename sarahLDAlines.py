from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet as wn
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize
import nltk
import string
import json
import scipy.sparse


#wfuDir=r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\wfuNews\articles.csv"
#stanDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\stanfordNews\articles.csv"
#dadeDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\MiamiDade\articles.csv"
#covidDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\covidArticles.csv"

#dfWFU = pd.read_csv(wfuDir, index_col=0, header=0)
#dfStan = pd.read_csv(stanDir, index_col=0, header=0)
#dfDade = pd.read_csv(dadeDir, index_col=0, header=0)
#dfCovid = pd.read_csv(covidDir, index_col=0, header=0)

#dfAll = pd.concat([dfStan, dfWFU, dfDade], ignore_index=True)

#preTwitterDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\JoshData\PreCovid.npz"
#preTwitterVocabDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\JoshData\PreVocab.txt"
#postTwitterDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\JoshData\PostCovid.npz"
#postTwitterVocabDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\JoshData\PostVocab.txt"

#textDirWFU = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\wfuNews\docWordMatrix.npz"
#vocabDirWFU = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\wfuNews\docWordVocab.txt"

#textDirStan = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\stanfordNews\docWordMatrix.npz"
#vocabDirStan = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\stanfordNews\docWordVocab.txt"

#textDirDade = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\MiamiDade\docWordMatrix.npz"
#vocabDirDade = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\MiamiDade\docWordVocab.txt"

saveDirPre = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\preCovidArticlesIndexed.csv"
saveDirPost = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\postCovidArticlesIndexed.csv"
dfPre = pd.read_csv(saveDirPre, index_col=0, header=0)
dfPost = pd.read_csv(saveDirPost, index_col=0, header=0)

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
            if word.find('.') != -1:
                sentenceCount += 1
            elif word.find('!') != -1:
                sentenceCount += 1
            elif word.find('?') != -1:
                sentenceCount += 1
    avgSentences = sentenceCount / docCount
    avgWord = wordCount / docCount
    print("Avg sentences: " + str(avgSentences))
    print("Avg words: " + str(avgWord))

#Get all articles that were written in January 2020 or later
def getCovidNews(df):
    preCovidIndices = []
    covidIndices = []
    dates = df.iloc[:, 1].values
    count = -1
    for date in dates:
        count += 1
        if type(date) != str:
            print("Found a nan")
        else:
            dateSplit = date.split('-')
            year = int(dateSplit[0])
            month = int(dateSplit[1])
            if year == 2020:
                if month >= 2:
                    covidIndices.append(count)
                    continue
            preCovidIndices.append(count)
            continue
    dfPostCovid = pd.DataFrame(index=np.arange(len(covidIndices)), columns=df.columns)
    count = -1
    for idx in covidIndices:
        count += 1
        dfPostCovid.iloc[count, 0] = df.iloc[idx, 0]
        dfPostCovid.iloc[count, 1] = df.iloc[idx, 1]
        dfPostCovid.iloc[count, 2] = df.iloc[idx, 2]
        dfPostCovid.iloc[count, 3] = df.iloc[idx, 3]

    dir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\postCovidArticles.csv"
    dfPostCovid.to_csv(dir)
    dfPreCovid = pd.DataFrame(index=np.arange(len(preCovidIndices)), columns=df.columns)
    count = -1
    for idx in preCovidIndices:
        count += 1
        dfPreCovid.iloc[count, 0] = df.iloc[idx, 0]
        dfPreCovid.iloc[count, 1] = df.iloc[idx, 1]
        dfPreCovid.iloc[count, 2] = df.iloc[idx, 2]
        dfPreCovid.iloc[count, 3] = df.iloc[idx, 3]

    dir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\preCovidArticles.csv"
    dfPreCovid.to_csv(dir)
    print('Transformed articles to pre- and post-covid\n')

#Remove any rows from dataframe with NANs
def removeNansFromDataFrame(df, saveDir):
    nanList = []
    text = df.iloc[:, 2].values
    for i in range(text.shape[0]):
        if type(text[i]) != str:
            nanList.append(i)
    print(nanList)
    for emptyIdx in nanList:
        print(df.iloc[emptyIdx-5:emptyIdx+5, 0:3])
        df = df.drop(emptyIdx, axis=0)
        df.reset_index(inplace=True, drop=True)
        print(df.iloc[emptyIdx-5:emptyIdx+5, 0:3])
    df.to_csv(saveDir)

#Lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, words):
        lemmaList = []
        for w in words:
            if len(wn.synsets(w)) != 0:
                tmp = wn.synsets(w)[0].pos()
                lemma = self.wnl.lemmatize(w, tmp)
                lemmaList.append(lemma)
            else:
                lemmaList.append(w)
        return lemmaList

#Gets stop words from nltk and also corpus specific stop words added by me
def getStopWords():
    stopWords = stopwords.words('english')
    stopWords = set(stopWords)
    for punc in string.punctuation:
        stopWords.add(punc)
    stopWords.add('one')
    stopWords.add('two')
    stopWords.add('three')
    stopWords.add('four')
    stopWords.add('five')
    stopWords.add('six')
    stopWords.add('seven')
    stopWords.add('eight')
    stopWords.add('nine')
    stopWords.add('get')
    stopWords.add('say')
    stopWords.add('would')
    stopWords.add('like')
    stopWords.add('seem')
    stopWords.add('also')
    stopWords.add('really')
    stopWords.add('com')
    stopWords.add('http')
    stopWords.add('org')
    stopWords.add('www')
    stopWords.add('wake')
    stopWords.add('forest')
    stopWords.add('stanford')
    stopWords.add('university')
    stopWords.add('campus')
    stopWords.add('quarter')
    stopWords.add('faculty')
    stopWords.add('mdc')
    stopWords.add('student')
    stopWords.add('take')
    stopWords.add('know')
    stopWords.add('year')
    stopWords.add('people')
    stopWords.add('make')
    stopWords.add('time')
    stopWords.add('show')
    stopWords.add('work')
    stopWords.add('undergraduate')
    stopWords.add('school')
    stopWords.add('miami')
    stopWords.add('want')
    stopWords.add('dade')
    stopWords.add('new')
    stopWords.add('many')
    return stopWords

stop_words = getStopWords()

#Function for preprocessing of data
def tokenize(text, stopWords):
    import re
    min_length = 3
    max_length = 18

    #Remove non-ascii characters from text
    byteText = str.encode(text, encoding='ascii', errors='ignore')
    text = byteText.decode('ascii', errors='strict')

    #Make text lowercase
    text = text.lower()

    #Get rid of punctuation
    punctuation = string.punctuation + '\n'
    text = re.sub('['+punctuation+']', ' ', text)
    text = re.sub(r'\d+', '', text)

    #Get rid of stopwords
    words = text.split()
    words = [word for word in words if word not in stopWords and len(word) >= min_length and len(word) <= max_length]

    #Lemmatize words
    lemmatizer = LemmaTokenizer()
    words = lemmatizer(words)

    #Stem words
    #stemmer = SnowballStemmer('english')
    #for word in words:
    #    stemmedWord = stemmer.stem(word)
    #    wordList.append(stemmedWord)

    #Ensure stemming/lemmatizing doesn't create words less than 3 length
    words = [word for word in words if len(word) >= min_length and word not in stopWords]
    return words

#Preprocess data and save it in a pandas DataFrame
def preProcessAndSave(df, textDir, stopWords):
    print("Starting preprocessing")
    allText = df.iloc[:, 2].values
    count = -1
    for text in allText:
        count += 1
        processedText = tokenize(text, stopWords)
        processedText = ' '.join(processedText)
        df.iloc[count, 2] = processedText
    df.to_csv(textDir)
    print("Wrote processed text to textDir\n")

saveDirPreProcessed = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\preCovidProcessedArticles.csv"
saveDirPostProcessed = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\postCovidProcessedArticles.csv"
dfPrePre = pd.read_csv(saveDirPreProcessed, index_col=0, header=0)
dfPrePost = pd.read_csv(saveDirPostProcessed, index_col=0, header=0)

#preProcessAndSave(dfPre, saveDirPreProcessed, stop_words)
#preProcessAndSave(dfPost, saveDirPostProcessed, stop_words)

#Return doc-word matrix (sklearn sparse matrix) from processed text - text is stored in a pandas DataFrame
def getDocWordMatrixandVocab(dfDir):
    print("Getting doc word matrix.")
    df = pd.read_csv(dfDir, header=0, index_col=0)
    #vocab = getVocabFromText(df)
    vectorizer = CountVectorizer(input='content',
                             lowercase=False,
                             max_df=0.95,
                             min_df=3)
    raw_text=df.iloc[:, 2].values
    docWordMatrix = vectorizer.fit_transform(raw_text)
    vocab = vectorizer.get_feature_names()
    print("Length of vocab: " + str(len(vocab)))
    return docWordMatrix, vocab

#Saves scipy sparse matrix and vocab to matrixDir and vocabDir, respectively
def saveMatrixAndVocab(matrix, vocab, matrixDir, vocabDir):
    scipy.sparse.save_npz(matrixDir, matrix)
    with open(vocabDir, 'w', encoding='utf-8', errors='ignore') as fp:
        for word in vocab:
            fp.write(word+'\n')
    fp.close()
    print("Saved sparse matrix and vocab.")

#preArticleMatrix, vocabPre = getDocWordMatrixandVocab(saveDirPreProcessed)
#postArticleMatrix, vocabPost = getDocWordMatrixandVocab(saveDirPostProcessed)

vocabPreDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\preArticlesVocab.txt"
vocabPostDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\postArticlesVocab.txt"
matrixPreDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\preArticles.npz"
matrixPostDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\postArticles.npz"

#saveMatrixAndVocab(preArticleMatrix, vocabPre, matrixPreDir, vocabPreDir)
#saveMatrixAndVocab(postArticleMatrix, vocabPost, matrixPostDir, vocabPostDir)

#Get vocabulary from a .txt file where vocab is stored as one word each line
def getVocabFromFile(vocabDir):
    vocab = []
    with open(vocabDir, 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp.readlines():
            vocab.append(line.strip('\n'))
    fp.close()
    return vocab

#Given directory where sparse matrix from count vectorizer is stored, returns sparse matrix
#Directory should end with .npz extension
def getSparseMatrix(dir):
    sparseMatrix = scipy.sparse.load_npz(dir)
    return sparseMatrix

#Returns vocab, sparse matrix from saved directories matrixDir and vocabDir
def getSparseMatrixandVocab(matrixDir, vocabDir):
    vocab = getVocabFromFile(vocabDir)
    matrix = getSparseMatrix(matrixDir)
    return matrix, vocab

preArticles, preArticlesVocab = getSparseMatrixandVocab(matrixPreDir, vocabPreDir)
postArticles, postArticlesVocab = getSparseMatrixandVocab(matrixPostDir, vocabPostDir)

#Takes in a scipy sparse matrix and tests for parameters alpha and beta
def evaluateParams(matrix, saveDir):
    print("Starting evaluation... this is gonna take a looooong time.\n")
    evalColumns = ["Num Topics", "Alpha", "Beta", "Perplexity"]
    dfEvaluation = pd.DataFrame(index=np.arange(4000), columns=evalColumns)
    count = -1
    for n_components in range(8, 28):
        for alpha in range(2, 22, 2):
            alpha = alpha / 100
            for beta in range(2, 42, 2):
                beta = beta / 100
                count += 1
                lda = LatentDirichletAllocation(n_components=n_components, doc_topic_prior=alpha, 
                                                topic_word_prior=beta)
                lda.fit(matrix)
                dfEvaluation.iloc[count, 0] = n_components
                dfEvaluation.iloc[count, 1] = alpha
                dfEvaluation.iloc[count, 2] = beta
                dfEvaluation.iloc[count, 3] = lda.perplexity(matrix)
        print("Finished eval of n_components number " + str(n_components))
    print("Finished eval for matrix")
    dfEvaluation.to_csv(saveDir)


def evaluateBetaAndAlpha(dir):
    dfEval = pd.read_csv(dir, index_col=0, header=0)
    print(dfEval.sort_values("Perplexity").groupby("Alpha", as_index=False).first())
    print(dfEval.sort_values("Perplexity").groupby("Beta", as_index=False).first())

import matplotlib.pyplot as plt

#Get plots of data
def horiz_plot(words, scores, fileBase, alpha, beta):
    #randomWords = ['bel', 'asdf', 'sdfsdf', 'sg b', 'asdfaeef', 'axserw', 'asdvvea', 'as', 'ab',
    #               'eaf', 'asfd', 'adsfee', 'vbda', 'asvneenl', 'safeee', 'ajajaja', 'bo', 'bp']
    rows = 6
    cols = 3
    numFigs = len(words)
    topicList = []
    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(30, 18)
    for i in range(rows):
        for j in range(cols):
            y_pos = np.arange(10)
            plotScores = scores[i*3 + j]
            plotWords = words[i*3 + j]
            axs[i, j].barh(y_pos, plotScores)
            axs[i, j].set_yticks(y_pos)
            axs[i, j].set_yticklabels(plotWords)
            for word in words[i*3 + j]:
                print(word)
            title = input("\nWhats the title? ")
            print('\n\n')
            #title = randomWords[i*3 + j]
            axs[i, j].set_title(title)
            topicList.append(title)
            axs[i, j].invert_yaxis()
    saveDir = fileBase + r"AllPlots" + str(numFigs) + "Topics" + str(alpha) + "Alpha" + str(beta) + "Beta.png"
    plt.savefig(saveDir)
    print("Saved figure.")
    plt.close()
    return topicList

#Run the model
def runModel(df, matrix, vocab, n_topics, alpha, beta, fileBase):
    lda = LatentDirichletAllocation(n_components=n_topics, 
                                    doc_topic_prior=alpha, 
                                    topic_word_prior=beta)
    lda.fit(matrix)

    n_top_words = 10
    topic_words = {}

    for topic, comp in enumerate(lda.components_):
        # for the n-dimensional array "arr":
        # argsort() returns a ranked n-dimensional array of arr, call it "ranked_array"
        # which contains the indices that would sort arr in a descending fashion
        # for the ith element in ranked_array, ranked_array[i] represents the index of the
        # element in arr that should be at the ith index in ranked_array
        # ex. arr = [3,7,1,0,3,6]
        # np.argsort(arr) -> [3, 2, 0, 4, 5, 1]
        # word_idx contains the indices in "topic" of the top num_top_words most relevant
        # to a given topic ... it is sorted ascending to begin with and then reversed (desc. now)
        word_idx = np.argsort(comp)[::-1][:n_top_words]
        # store the words most relevant to the topic
        topic_words[topic] = [(vocab[i], comp[i]) for i in word_idx]
    
    wordList = []
    scoreList = []
    topicCount = -1
    for topic, words in topic_words.items():
        topicCount += 1
        wordList.append([])
        scoreList.append([])
        #print('Topic: %d' % (topic + 1))
        #print('  %s' % ', '.join(words))
        for word in words:
            wordList[topicCount].append(word[0])
            scoreList[topicCount].append(word[1])
    topics = horiz_plot(wordList, scoreList, fileBase, alpha, beta)
    setTopics = set()
    for word in topics:
        setTopics.add(word)
    setTopics.add('unknown')
    listOfTopics = list(setTopics)

    df['Topic Number'] = 0
    X = lda.transform(matrix)

    for i in range(len(X)):
        if X[i].max() == X[i].min():
            df.iloc[i, 4] = len(listOfTopics) - 1
            continue
        else:
            count = -1
            for j in X[i]:
                count += 1
                if j == X[i].max():
                    topicToGet = topics[count]
                    topicIdx = listOfTopics.index(topicToGet)
                    df.iloc[i, 4] = topicIdx
    return df, listOfTopics


def timeHistogram(df, topics, saveDirBase):
    import seaborn as sns
    numTopics = len(set(df.iloc[:, 4].values))

    count = -1
    for date in df.iloc[:, 1].values:
        count += 1
        dateSplit = date.split('-')
        if dateSplit[1].find('0') == -1 and len(dateSplit[1]) != 2:
            newDate = dateSplit[0] + '-' + '0' + dateSplit[1]
            df.iloc[count, 1] = newDate
        else:
            df.iloc[count, 1] = dateSplit[0] + '-' + dateSplit[1]

    dateSet = set()
    dates = df.iloc[:, 1].values
    for date in dates:
        dateSet.add(date)
    dateList = list(dateSet)

    articleCounts = dict()
    for j in dateList:
        articleCount = df.loc[df['Year'] == j].shape[0]
        articleCounts[j] = articleCount
    datesAndTopicCounts = []
    for i in range(numTopics):
        for j in dateList:
            dfSingle = df.loc[(df['Year'] == j) & (df['Topic Number'] == i)]
            proportion = dfSingle.shape[0]/articleCounts[j]
            datesAndTopicCounts.append([j, topics[i], proportion])
    dfHistogram = pd.DataFrame(data=datesAndTopicCounts, columns = ['Date', 'Topic', 'Percentage'])
    
    dfHistogram = dfHistogram.sort_values(by=['Date'])

    for i in range(numTopics):
        dfLocal = dfHistogram.loc[dfHistogram['Topic'] == topics[i]]
        fig, ax = plt.subplots(figsize=(20, 12))
        g = sns.barplot(x=dfLocal.iloc[:, 0], y=dfLocal.iloc[:, 2], ax=ax)
        rowCount = -1
        for index, row in dfLocal.iterrows():
            rowCount += 1
            if row.Percentage != 0:
                strPercent = str(round(row.Percentage*100, 1)) + r'%'
                g.text(rowCount, row.Percentage, strPercent, color='black', ha="center")
        title = "Ratio of " + dfLocal.iloc[0,1] + " articles out of all articles per month"
        plt.title(title)
        plt.xticks(rotation=45)
        topicName = dfLocal.iloc[0, 1]
        if topicName.find('/') != -1:
            topicName = topicName.replace('/', '-')
        plt.savefig(saveDirBase + "\\" + topicName + '.png')
        #plt.show()
        plt.close()
    print("Saved time plots.")


def analyzeSentimentPerTopic(df, topics):
    import tabulate
    #nltk.download('vader_lexicon')
    from nltk.corpus import subjectivity
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
    #df['Sentiment'] = {'compound': 0, 'neg': 0, 'neu':0, 'pos':0}
    dfSentiment = pd.DataFrame(data=np.zeros(shape=(len(topics), 5)), index=np.arange(len(topics)), columns=['Topic', 'compound', 'neg', 'neu', 'pos'])
    for i in range(len(topics)):
        compound = 0; neg = 0; neu = 0; pos = 0
        dfLocal = df.loc[df['Topic Number'] == i]
        for article in dfLocal.iloc[:, 2].values:
            if type(article) != float:
                ss = sid.polarity_scores(article)
                compound += ss['compound']; neg += ss['neg']
                neu += ss['neu']; pos += ss['pos']
        dfSentiment.iloc[i, 0] = topics[i]
        if compound != 0:
            dfSentiment.iloc[i, 1] = compound/dfLocal.shape[0]
        if neg != 0:
            dfSentiment.iloc[i, 2] = neg/dfLocal.shape[0]
        if neu != 0:
            dfSentiment.iloc[i, 3] = neu/dfLocal.shape[0]
        if pos != 0:
            dfSentiment.iloc[i, 4] = pos/dfLocal.shape[0]
    print(tabulate.tabulate(dfSentiment, headers=dfSentiment.columns))
    print()





#imageDirTwitter = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\JoshData\Visualizations"
#postImageDirTwitter = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\JoshData\Visualizations"

#runModel(preTwitter, preTwitterVocab, 18, 0.001, 0.4, (imageDirTwitter + r"\pre"))
#runModel(postTwitter, postTwitterVocab, 18, 0.001, 0.4, (imageDirTwitter + r"\post"))


imageDir = r"C:\Users\andre\Desktop\Data Mining SUMMER RESEARCH\Topic Modeling Project\data\covidNews\Visualizations"

dfTopics, topics = runModel(dfPrePre, preArticles, preArticlesVocab, 18, 0.01, 0.28, (imageDir + r"\pre"))
timeHistogram(dfTopics, topics, imageDir + r'\TopicsByDatePreCovid')
analyzeSentimentPerTopic(dfTopics, topics)

dfTopics, topics = runModel(dfPost, postArticles, postArticlesVocab, 18, 0.01, 0.28, (imageDir + r"\post"))
timeHistogram(dfTopics, topics, imageDir + r'\TopicsByDatePostCovid')
analyzeSentimentPerTopic(dfTopics, topics)


############## UNHELPFUL METHODS ###############

#Get vocabulary from processed text for use in CountVectorizer
def getVocabFromText(df):
    raw_text = df.iloc[:, 2].values
    vocabSet = set()
    for text in raw_text:
        for word in text.split():
            vocabSet.add(word)
    vocabList = list(vocabSet)
    return vocabList


def getRandIndexMatrix(df, topics):
    
    for i in range(len(topics)):
        for j in range(len(topics)):
            j += 1
            dfTopic1 = df.loc[df['Topic Number'] == i]
            dfTopic2 = df.loc[df['Topic Number'] == j]
            raw_docs1 = dfTopic1.iloc[:, 2].values
            raw_docs2 = dfTopic2.iloc[:, 2].values
            randMatrix = np.zeros((len(raw_docs1), len(raw_docs2)))
            count1 = -1
            for doc1 in raw_docs1:
                count1 += 1
                count2 = -1
                for doc2 in raw_docs2:
                    count2 += 1
                    score = 0
                    listDoc1 = doc1.split()
                    listDoc2 = doc2.split()
                    setDoc1 = set(listDoc1)
                    setDoc2 = set(listDoc2)
                    intersection = setDoc1.intersection(setDoc2)
                    randMatrix[count1, count2] = len(intersection)
            print(randMatrix)