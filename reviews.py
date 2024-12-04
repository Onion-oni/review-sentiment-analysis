#Importing libraries
import pandas as pd
import numpy as np
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

#importing dataset csv
df = pd.read_csv('netflix_fb_comments.csv')

df['SID Scores'] = df['Comment'].apply(lambda review: sid.polarity_scores(review))
df['Compound'] = df['SID Scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c > 0 else 'neg')

