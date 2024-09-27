# -*- coding: utf-8 -*-

# this code does the pre-processing of tweets based on hashtags
#-> pops out unwanted data,removes sybmbols and punctuations
# -> gets all the hashtags used 
#-> segregates hashtags based on custom-made dictionary for anti and pro climate change
# -> does sentiment analysis on sorted tweets having only anti and pro hashtags
#->from line 149 to 153: it plots a bar graph showing occurance of hashtags
#-> on line 74 and 75 there is list (dictionary), we have to update the list with more buzzwords. ..
#I have written buzz words which I managed to get from tweets. [distribution is skewed as of now]
# Next thing will be label the sorted tweets as -1 to +1 [Pending] [made a df which has senti score of vader & blob]
# Train the model using LSTM [Pending]
# Compare it with baseline models like vader/textblob etc [Pending]

# space issue [solved by word segmenter]
# remove duplicates
#not cleaned properly
#csv files are mode for tweeter 2023 and maston climate.
#mastadom energy is more of news. so should be neglected as we did for tweeter2022
#have tweaked values of some words in vader as thoses words are positive in our contenxt. such as 
# "crisis,emergency "-> shows conerns of pro-climate chnage people
# vader is more accurate than textblob. we can use hybrid approach as well
# we might need to do some random sampleing of tweets for its positive or negative sentiment
"""
Created on Sat Oct  7 10:40:24 2023

@author: Akrosh
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordsegment import load, segment
load()
import re
import wordninja
from textblob import TextBlob, Word, Blobber

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 

def sep_hashtags(m):
    s=m.group(1)
    return ' '.join(re.split(r'(?=[A-Z])', s))

df=pd.read_csv("C:/Users\Akrosh\OneDrive - rit.edu\Desktop\Data Visualization/twitter_climate_2023.csv")
df.drop_duplicates(subset='text',inplace=True,keep='first')
df.info()

content_df=df['text']

########3
df.pop("id") 
df.pop("user_name") 
df.pop("user_description") 
df.pop("user_location") 
df.pop("user_following") 
df.pop("user_followers") 
df.pop("tweets_by_user") 
df.pop("user_created_at") 
df.pop("tweet_created") 
df.pop("retweets") 
df.pop("likes") 
df.pop("media") 
df.pop("search_term") 
df.pop("text") 
df2=df["hashtags"] 


#Removing punctuations
df3=df2.to_frame("hashtags")
Lists_no_sym=[]
tokenized_list=[]

#tokenizing and removing symbols and lowering the cases
for i in range (0,len(df3)):
    #no_symbol=re.sub(r"[^A-Za-z0-9\s]+", "", df3.iloc[i,0])
    no_symbol=re.sub(r"[^A-Za-z0-9\s]+", " ", df3.iloc[i,0])
    #[^\x00-\x7F]+
    Lists_no_sym.append(no_symbol)
print(Lists_no_sym[4])
for i in range(0,len(Lists_no_sym)):
    tokenized=nltk.word_tokenize(Lists_no_sym[i])
    tokenized_list.append(tokenized)

hashtags1=[[] for l in range(0,len(tokenized_list))]
#hashtags1=[]

#get the hashtags after the word "name" and segments each multiword hashtag into its single word 
for j in range(0,len(tokenized_list)):
    result = [i for i, x in enumerate(tokenized_list[j]) if x == 'text']
    for k in result:
        if k+1<len(tokenized_list[j]):
            hashtags1[j].append(' '.join(map(str,segment(tokenized_list[j][k+1]))))
    #del(result)
    print ("analysing row : ",j,"out of ",len(tokenized_list))


#graph for hastags frequency
total_hashtags = sum(hashtags1, [])
frequency = nltk.FreqDist(total_hashtags)
hashtag_df = pd.DataFrame({'hashtag': list(frequency.keys()),'count': list(frequency.values())})
hashtag_df = hashtag_df.nlargest(103803, columns="count")
hashtag_df.plot.bar(x="hashtag",y="count",title="most used hashtags")
savefile=hashtag_df.to_csv('Twitter 2023 hashtags.csv')


## ## Part is complete########################################################### this part is complete####


pro=['climate justice','environmental justice', 'climate justice now', 'climate action','climate action now','climate change action', 'climate crisis', 'climate emergency']
anti=['climate hoax','hoax','climate change hoax','climate hysteria', 'climate scam','climate change scam', 'global warming scam', 'carbon tax scam', 'climate cult', 'agw hoax', 'climate alarmist']


#dictionary for anti and pro believers [to be done for twitter]

pro_index=[]
anti_index=[]
comb_index=[]
for i in range(0,len(hashtags1)):
    if bool(set(hashtags1[i]).intersection(set(pro))):
        pro_index.append(i)
        comb_index.append(i)
    if bool(set(hashtags1[i]).intersection(set(anti))):
        anti_index.append(i)
        comb_index.append(i)

plt.figure(1)
plt.pie([len(pro_index), len(anti_index)],labels=["Pro hashtags", "anti hashtags"])

comb_index2=[]
[comb_index2.append(i) for i in comb_index if i not in comb_index2]


##############content segment...........................

content_dfv1=content_df.to_frame("text")
content_dfv2=[]
for i in range(0,len(comb_index2)):
    content_dfv2.append(content_dfv1.iloc[comb_index2[i],0])
#list to df
content_dfv3=pd.DataFrame()
content_dfv3["text"]=content_dfv2
               
Lists_no_sym_content=[]

#removing text between " "
for i in range(0,len(content_dfv3)):
    #text_only = re.sub(r'("[^"]*)"', " ", content_dfv3.iloc[i,0])
    text_only = re.sub(r'(?:\@|https?\://)\S+', " ", content_dfv3.iloc[i,0])
    text_only=re.sub(r'#(\S+)',sep_hashtags,text_only)
    tokenized = nltk.word_tokenize(text_only)
    no_symbol_content=[word.lower() for word in tokenized if word.isalpha()]
    Lists_no_sym_content.append(no_symbol_content)

non_essential_words=list(stopwords.words('english'))

non_essential_words.append('p')
non_essential_words.append('https')
non_essential_words.append('hashtag')
non_essential_words.append('nofollow')
non_essential_words.append('mention')
non_essential_words.append('noreferrer')
non_essential_words.append('noopener')
non_essential_words.append('span')
non_essential_words.append('tag')
non_essential_words.append('amp')
non_essential_words.append('invisible')
non_essential_words.append('ellipsis')
non_essential_words.append('br')

newl=[]
z=0;

essential_word=[]
while z<len(Lists_no_sym_content):
    newl=[]
    for i in Lists_no_sym_content[z]:
        if i not in non_essential_words:
            newl.append(i)
    essential_word.append(' '.join(newl))
    del(newl)
    z=z+1

## separte words
dumy_var=[]


for i in range(0,len(essential_word)): # will add it once i have gpu 
    dumy_var.append(wordninja.split(essential_word[i]))
    #dumy_var.append(segment(essential_word[i]))
    print(i)
essential_word2=[]
for i in range(0,len(dumy_var)):
    essential_word2.append(' '.join(dumy_var[i]))    


#DF with refined/essential list of words
content_dfv3["content with essential words"]=essential_word
  
content_dfv3=content_dfv3.drop_duplicates(subset='content with essential words',inplace=False,ignore_index=True)
#content_dfv3.dropna(inplace=True)

#content_dfv3=[]
#content_dfv3=xx
#sentiment analysis of tweets (anti and pro hastags)
vader_sentiment=[]
textblob_sentiment=[]
senti=SentimentIntensityAnalyzer()
### to update vader lexicon

vader_new_words_df = pd.read_excel('C:/Users\Akrosh\OneDrive - rit.edu\Desktop\Data Visualization/New Words for VADER.xlsx')
vader_new_words_df['words']=vader_new_words_df['words'].str.split().str.join(' ')
vader_new_words_df2=vader_new_words_df.drop_duplicates(subset='words')
new_words=vader_new_words_df2.set_index('words').to_dict()['Scores']

#new_words = {
 #   'crisis': 4.0,
  #  'emergency': 4.0,
   # 'catastrophe': 4.0
#}

senti.lexicon.update(new_words)
sentiment=[]
for i in range(0,len(content_dfv3)):
 vader_sentiment.append(senti.polarity_scores(content_dfv3.iloc[i][1]))
 textblob_sentiment.append(TextBlob(content_dfv3.iloc[i][1]))
  

df5=pd.DataFrame(vader_sentiment)
score_thresh=0
content_dfv3[["Vader_-ve_score", "Vader_neu_score","Vader_+ve_score","Vader_comp_score"]]=df5[["neg","neu","pos","compound"]]
#content_dfv3['textblob_score']=textblob_sentiment #to be uncommented
content_dfv3['Vader Sentiment Label']= ''
content_dfv3.loc[content_dfv3['Vader_comp_score']>-score_thresh,'Vader Sentiment Label']='Pro-Climate'
content_dfv3.loc[(content_dfv3['Vader_comp_score']<=score_thresh) & (content_dfv3['Vader_comp_score']>=-score_thresh) ,'Vader Sentiment Label']='Neutral'
content_dfv3.loc[content_dfv3['Vader_comp_score']<-score_thresh,'Vader Sentiment Label']='Anti-Climate'
# for indepth classification
content_dfv3['Vader Indepth Sentiment Label']= ''
content_dfv3.loc[(content_dfv3['Vader_comp_score']>0) & (content_dfv3['Vader_comp_score']<=0.5),'Vader Indepth Sentiment Label']='Moderate Pro-Climate'
content_dfv3.loc[(content_dfv3['Vader_comp_score']>0.5) & (content_dfv3['Vader_comp_score']<=1),'Vader Indepth Sentiment Label']='Extreme Pro-Climate'
content_dfv3.loc[(content_dfv3['Vader_comp_score']==0),'Vader Indepth Sentiment Label']='Neutral'
content_dfv3.loc[(content_dfv3['Vader_comp_score']<0) & (content_dfv3['Vader_comp_score']>=-0.5) ,'Vader Indepth Sentiment Label']='Moderate Anti-Climate'
content_dfv3.loc[(content_dfv3['Vader_comp_score']<-0.5) & (content_dfv3['Vader_comp_score']>=-1) ,'Vader Indepth Sentiment Label']='Extreme Anti-Climate'
#graph to show the portion of anti, pro and neutral sentiment (Vader)
plt.figure(2)
content_dfv3[['Vader Sentiment Label']].value_counts().plot(kind='bar',title='Vader Sentiment Analysis')


#graph to show the portion of anti, pro and neutral sentiment (TextBlob) to be made
content_dfv3['textblob_sent']=[textblob_sentiment[i].sentiment.polarity for i in range(0, len(content_dfv3))]
content_dfv3['textblob Sentiment Label']= ''
content_dfv3.loc[content_dfv3['textblob_sent']>-score_thresh,'textblob Sentiment Label']='Pro-Climate'
content_dfv3.loc[(content_dfv3['textblob_sent']<=score_thresh) & (content_dfv3['textblob_sent']>=-score_thresh),'textblob Sentiment Label']='Neutral'
content_dfv3.loc[content_dfv3['textblob_sent']<-score_thresh,'textblob Sentiment Label']='Anti-Climate'


content_dfv3['textblob Indepth Sentiment Label']= ''
content_dfv3.loc[(content_dfv3['textblob_sent']>0) & (content_dfv3['textblob_sent']<=0.5),'textblob Indepth Sentiment Label']='Moderate Pro-Climate'
content_dfv3.loc[(content_dfv3['textblob_sent']>0.5) & (content_dfv3['textblob_sent']<=1),'textblob Indepth Sentiment Label']='Extreme Pro-Climate'
content_dfv3.loc[(content_dfv3['textblob_sent']==0),'textblob Indepth Sentiment Label']='Neutral'
content_dfv3.loc[(content_dfv3['textblob_sent']<0) & (content_dfv3['textblob_sent']>=-0.5) ,'textblob Indepth Sentiment Label']='Moderate Anti-Climate'
content_dfv3.loc[(content_dfv3['textblob_sent']<-0.5) & (content_dfv3['textblob_sent']>=-1) ,'textblob Indepth Sentiment Label']='Extreme Anti-Climate'

plt.figure(3)
content_dfv3[['textblob Sentiment Label']].value_counts().plot(kind='bar',title='textblob Sentiment Analysis')


from statistics import mean
content_dfv3['mean sentiment']=''
content_dfv3['mean sentiment']=[(content_dfv3.iloc[i,5]+content_dfv3.iloc[i,8])/2 for i in range(0, len(content_dfv3))]

content_dfv3['Combined (vader and textblob Sentiment Label']= ''
#content_dfv3.loc[content_dfv3['mean sentiment']>0,'Combined (vader and textblob Sentiment Label']='Pro-Climate'
#content_dfv3.loc[content_dfv3['mean sentiment']<=0,'Combined (vader and textblob Sentiment Label']='Neutral'
#content_dfv3.loc[content_dfv3['mean sentiment']<0,'Combined (vader and textblob Sentiment Label']='Anti-Climate'
content_dfv3.loc[content_dfv3['mean sentiment']>0,'Combined (vader and textblob Sentiment Label']='Pro-Climate'
content_dfv3.loc[content_dfv3['mean sentiment']<=0,'Combined (vader and textblob Sentiment Label']='Neutral'
content_dfv3.loc[content_dfv3['mean sentiment']<0,'Combined (vader and textblob Sentiment Label']='Anti-Climate'

content_dfv3['Combined Indepth (vader and textblob Sentiment Label']= ''
content_dfv3.loc[(content_dfv3['mean sentiment']>0) & (content_dfv3['mean sentiment']<=0.5),'Combined Indepth (vader and textblob Sentiment Label']='Moderate Pro-Climate'
content_dfv3.loc[(content_dfv3['mean sentiment']>0.5) & (content_dfv3['mean sentiment']<=1),'Combined Indepth (vader and textblob Sentiment Label']='Extreme Pro-Climate'
content_dfv3.loc[(content_dfv3['mean sentiment']==0),'Combined Indepth (vader and textblob Sentiment Label']='Neutral'
content_dfv3.loc[(content_dfv3['mean sentiment']<0) & (content_dfv3['mean sentiment']>=-0.5) ,'Combined Indepth (vader and textblob Sentiment Label']='Moderate Anti-Climate'
content_dfv3.loc[(content_dfv3['mean sentiment']<-0.5) & (content_dfv3['mean sentiment']>=-1) ,'Combined Indepth (vader and textblob Sentiment Label']='Extreme Anti-Climate'

#Printing the stats of vader and textblob

print('The number of pro-climate as per Vader is :',len(content_dfv3[content_dfv3['Vader Sentiment Label']=='Pro-Climate']),'against hpothesized value of',len(pro_index))
print('The number of anti-climate as per Vader is :',len(content_dfv3[content_dfv3['Vader Sentiment Label']=='Anti-Climate']),'against hpothesized value of',len(anti_index))
print('The number of neutral as per Vader is :',len(content_dfv3[content_dfv3['Vader Sentiment Label']=='Neutral']),'against hpothesized value of 0 \n\n')

print('The number of pro-climate as per textblob is :',len(content_dfv3[content_dfv3['textblob Sentiment Label']=='Pro-Climate']),'against hpothesized value of',len(pro_index))
print('The number of anti-climate as per textblob is :',len(content_dfv3[content_dfv3['textblob Sentiment Label']=='Anti-Climate']),'against hpothesized value of',len(anti_index))
print('The number of neutral as per textblob is :',len(content_dfv3[content_dfv3['textblob Sentiment Label']=='Neutral']),'against hpothesized value of 0\n\n')

print('The number of pro-climate as per combined values is :',len(content_dfv3[content_dfv3['Combined (vader and textblob Sentiment Label']=='Pro-Climate']),'against hpothesized value of',len(pro_index))
print('The number of anti-climate as per combined value is :',len(content_dfv3[content_dfv3['Combined (vader and textblob Sentiment Label']=='Anti-Climate']),'against hpothesized value of',len(anti_index))
print('The number of neutral as per combined value is :',len(content_dfv3[content_dfv3['Combined (vader and textblob Sentiment Label']=='Neutral']),'against hpothesized value of 0\n\n')


plt.figure(5)
content_dfv3[['Combined (vader and textblob Sentiment Label']].value_counts().plot(kind='bar',title='Hybrid Sentiment Analysis') 

#save csv file for lstm
df_lstm=pd.DataFrame()
df_lstm=content_dfv3
df_lstm.to_csv('LSTMTwitterData2023.csv')


# plotting common words twitter 2023 data
commonwords=10;
words_in_tweet = sum(dumy_var,[])
frequency_word = nltk.FreqDist(words_in_tweet)
word_df = pd.DataFrame({'word': list(frequency_word.keys()),'count': list(frequency_word.values())})
word_df = word_df.nlargest(commonwords, columns="count")
plt.figure(6)
word_df.plot.bar(x="word",y="count",title="most used word")
save_commonword=word_df.to_csv('commonwords.csv')
##########














