#!/usr/bin/env python
# coding: utf-8

# In[58]:


#!pip install textblob


# In[32]:


import numpy as np 
import pandas as pd
import re
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
from textblob import TextBlob
from PIL import Image
warnings.simplefilter("ignore")


# ## Data Manipulation and Data Exploration
# 

# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/Smurkio8/FinalProject720/main/rafaelnadal_tweets.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# ### Missing Values

# In[5]:


data= df[df['user_location'].notna()]
data.shape


# In[6]:


def missing_data(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[7]:


missing_data(df)


# ### Unique Values
# 

# In[9]:


def unique_values(df):
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return(np.transpose(tt))


# In[10]:


unique_values(df)


# ### Rename of Columns

# In[14]:


df.columns = ['Name', 'Location', 'Description', 'Created_in','Followers', 'Friends', 'Favourites','User_verified', 'Date', 'Text', 'Hashtags', 'Source', 'Is_retweet']
df.head(2)


# ### Most Frequent Values 

# In[15]:


def most_frequent_values(df):
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in df.columns:
        itm = df[col].value_counts().index[0]
        val = df[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))


# In[16]:


most_frequent_values(df)


# In[17]:


print(f"data shape: {df.shape}")


# In[18]:


df.describe()


# ## Data Visualization

# In[19]:


def plot_count(feature, title, df, size=1, ordered=True):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    if ordered:
        g = sns.countplot(x=feature, data=df, order = df[feature].value_counts().index[:20], palette='Set3')
    else:
        g = sns.countplot(x=feature, data=df, palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# ### User Name

# In[20]:


plot_count("Name", "count", df,4)


# ### User Location

# In[21]:


plot_count("Location", "Count", df,4)


# ### Tweet Source

# In[22]:


plot_count("Source", "count", df,4)


# ### Text Wordcloauds

# In[23]:


mask = 255 - np.array(Image.open('tennis-ball-.png'))


# In[53]:


def show_wordcloud(data, title="", mask=None, color="white"):
    text = " ".join(t for t in data.dropna())
    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U", "Rafa", "rafaelnadal𓃵","Rafaelnadal", "Nadal", "FrenchOpen","RolandGarros" ,"FrenchOpen2022", "French", "Open", "2022", "tenni", "GOAT"])
    wordcloud = WordCloud(stopwords=stopwords, scale=2, max_font_size=50, max_words=500,mask=mask,background_color=color).generate(text)
    fig = plt.figure(1, figsize=(15,15))
    plt.axis('off')
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=1.0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()


# In[54]:


show_wordcloud(df['Text'], title = 'Prevalent words in tweets', mask=mask)


# In[55]:


usa_df = df.loc[df.Location=="United States"]
show_wordcloud(usa_df['Text'], title = 'Prevalent words in tweets from USA', mask=mask)


# ### Top Hashtags

# In[63]:


hashtags = []
for tweet in df.Text:
    hashtag = re.findall(r"#(\w+)", tweet)
    hashtags.extend(hashtag)

# Count the frequency of each hashtag and plot the top 10
plt.figure(figsize=(5, 4))
top_hashtags = pd.Series(hashtags).value_counts().head(10)
top_hashtags.plot(kind='bar')
plt.title("Top 10 Hashtags")
plt.xlabel("Hashtags")
plt.ylabel("Frequency")
plt.show()


# In[26]:


#!pip install nltk
import nltk
nltk.download('vader_lexicon')


# ## Sentimental Analysis

# In[27]:


warnings.simplefilter("ignore")
sia = SentimentIntensityAnalyzer()
def find_sentiment(post):
    try:
        if sia.polarity_scores(post)["compound"] > 0:
            return "Positive"
        elif sia.polarity_scores(post)["compound"] < 0:
            return "Negative"
        else:
            return "Neutral"  
    except:
        return "Neutral"


# In[28]:


def plot_sentiment(df, feature, title):
    counts = df[feature].value_counts()
    percent = counts/sum(counts)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    counts.plot(kind='bar', ax=ax1, color='green')
    percent.plot(kind='bar', ax=ax2, color='blue')
    ax1.set_ylabel(f'Counts : {title} sentiments', size=12)
    ax2.set_ylabel(f'Percentage : {title} sentiments', size=12)
    plt.suptitle(f"Sentiment analysis: {title}")
    plt.tight_layout()
    plt.show()


# In[29]:


df['text_sentiment'] = df['Text'].apply(lambda x: find_sentiment(x))
plot_sentiment(df, 'text_sentiment', 'Text')

