import contractions
from bs4 import BeautifulSoup
import unicodedata
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import numpy as np
import pandas as pd
import re



def LowerCase(df):
    df['text']=df['text'].apply(lambda x:str(x).lower())
    return df

#contraction to expansion  (I'll: I Will)
def cont2expansion(df):
    def Cont2Exp(x):
        if type(x) is str:
            a=[]
            for word in x.split():
                a.append(contractions.fix(word))
            expanded_text = ' '.join(a)
            return expanded_text
        else:
            return x
    df['text']=df['text'].apply(lambda x:Cont2Exp(x))
    return df

#Removing Emails from the dataset
def Remove_Emails(df):
    import re
    df['text']=df['text'].apply(lambda x:re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+._-]+)',"",x))
    return df

#removing urls
def Remove_urls(df):
    import re
    df['text']=df['text'].apply(lambda x:(re.sub(r'(http|https|ftp|ssh://([\w_-]+(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',x)))
    return df

#remove rt from the begining
def Remove_rt(df):
    import re
    df['text']=df['text'].apply(lambda x:(re.sub(r'\brt\b','',x)))
    return df


#Remove Special Characters
def Remove_SpecialChar(df):
    import re
    df['text']=df['text'].apply(lambda x:re.sub(r'[^\w ]+',"",x))
    return df

#Remove Numeric
def Remove_Numeric(df):
    import re
    df['text']=df['text'].apply(lambda x:re.sub(r'[0-9]','',x))
    return df

#Remove Extra Spaces
def Remove_ExtraSpaces(df):
    df['text']=df['text'].apply(lambda x:' '.join(x.split()))
    return df

#Remove HTML Tags
def Remove_HTMLTags(df):
    df['text']=df['text'].apply(lambda x:BeautifulSoup(x, 'lxml').get_text().strip())
    return df

# Remove Accented Char
def Remove_AccentedChar(df):
    import unicodedata
    def RemoveAccentedChar(x):
        x=unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore')
        return x
    df['text']=df['text'].apply(lambda x:RemoveAccentedChar(x))
    return df

#Removing StopWords
def Remove_StopWords(df):
    df['text']=df['text'].apply(lambda x: ' '.join([t for t in x.split() if t not in STOP_WORDS]))
    return df
    
# Covert text to it's base form (Lemmatizing)
def Convert2Base(df):
    import spacy
    nlp=spacy.load('en_core_web_sm')
    def make_to_base(x):
        x=str(x)
        x_list=[]
        doc=nlp(x)

        for token in doc:
            lemma=token.lemma_
            if lemma=='-PRON-' or lemma=='be':
                lemma=token.text
            x_list.append(lemma)
        return ' '.join(x_list)
    df['text']=df['text'].apply(lambda x:make_to_base(x))
    return df

#Removing Most occuring words
def Remove_MostOccuring(df):
    text=' '.join(df['text'])
    text=text.split()
    freq_comm=pd.Series(text).value_counts()
    f2=freq_comm[:2]
    df['text']=df['text'].apply(lambda x:' '.join([t for t in x.split() if t not in f2]))
    return df

#Removing Rarely occuring words
def Remove_RarelyOccuring(df):
    text=' '.join(df['text'])
    text=text.split()
    freq_comm=pd.Series(text).value_counts()
    rare10000=freq_comm.tail(10000)
    df['text']=df['text'].apply(lambda x:' '.join([t for t in x.split() if t not in rare10000]))
    return df

# Remove Blank tweet 
def Remove_Blank(df):
    a=df['text'].apply(lambda x:len(x.split()))
    df_filtered=df[a==0]
    df.drop(df_filtered.index, inplace = True)
    return df


