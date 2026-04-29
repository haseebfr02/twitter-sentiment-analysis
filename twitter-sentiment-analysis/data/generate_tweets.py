"""
Twitter Sentiment Dataset Generator
=====================================
Generates 2,000 realistic tweets across 5 topics
with realistic sentiment distribution.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── Tweet Templates ───────────────────────────────────────────
TOPICS = ['AI Technology', 'Climate Change', 'Economy', 'Sports', 'Social Media']

POSITIVE_TWEETS = [
    "Just tried the new AI tool and it's absolutely amazing! 🚀 #AI #Tech",
    "Loving how technology is making our lives easier every day! #Innovation",
    "Great news for the economy! Jobs are up and things are looking bright 💪",
    "What an incredible game last night! Our team is unstoppable! 🏆 #Sports",
    "Social media has connected me with so many wonderful people worldwide ❤️",
    "This new AI feature is a game changer for productivity! #MachineLearning",
    "Renewable energy is the future and it's happening now! 🌱 #ClimateAction",
    "Record breaking performance today! So proud of our athletes 🥇",
    "The market rally today is incredible! Economy bouncing back strong 📈",
    "Twitter is such a great platform to share ideas and connect! #SocialMedia",
    "AI is transforming healthcare in ways we never imagined! #HealthTech",
    "Our team just won the championship! Best feeling ever! 🎉 #Champions",
    "New climate policy is a step in the right direction! #GreenFuture",
    "Entrepreneurship is booming! So many exciting startups right now 💼",
    "Just discovered an amazing community on social media! So supportive ✨",
    "The future of AI looks incredibly promising! Excited for what's next 🤖",
    "Beautiful day to think about how far we've come in renewable energy 🌞",
    "Small businesses are thriving! Love seeing local entrepreneurs succeed",
    "That comeback victory was unbelievable! Best sports moment of the year!",
    "Connected with inspiring people on Twitter today. This platform rocks!",
]

NEGATIVE_TWEETS = [
    "AI is going to take all our jobs. This is terrifying 😰 #AIRisk",
    "Climate change is getting worse and nobody seems to care enough 😢",
    "Inflation is killing the middle class. Can't afford basic necessities anymore",
    "Terrible refereeing in tonight's game. Complete disaster 😤 #Sports",
    "Social media is destroying mental health especially for young people 😔",
    "AI companies are collecting our data without real consent. Scary stuff",
    "We are running out of time to fix climate change. Politicians do nothing",
    "Stock market crash wiping out retirement savings. This is a nightmare 📉",
    "That performance was embarrassing. Worst game I've watched in years 😤",
    "Twitter has become such a toxic place. Full of hate and misinformation",
    "AI bias is a serious problem and companies aren't doing enough about it",
    "Another climate disaster and world leaders still dragging their feet",
    "Unemployment rising again. Economy is in terrible shape right now",
    "Injured our best player. Season might be over before it started 😭",
    "Cyberbullying on social media is getting completely out of control",
    "Deepfakes powered by AI are threatening democracy itself. Wake up people",
    "Wildfires getting worse every year. Climate crisis is here NOW",
    "Recession fears growing. Small businesses closing left and right 😞",
    "That trade was a disaster. Management has no idea what they're doing",
    "Social media algorithms are designed to make us angry. It's working",
]

NEUTRAL_TWEETS = [
    "New AI regulations being discussed in parliament today #Policy #AI",
    "Climate summit scheduled for next month. World leaders to attend",
    "Federal Reserve announces new interest rate decision next week",
    "Transfer window closes tomorrow. Several deals still being negotiated",
    "Social media usage statistics released for Q1 2026 #Report",
    "AI startup raises Series B funding round. Details to follow",
    "Annual climate report published today. Data shows mixed results",
    "GDP figures for last quarter released by government statistics office",
    "League standings updated after this weekend's fixtures #Football",
    "New social media platform launched targeting Gen Z users today",
    "OpenAI announces new research paper on language models",
    "UN climate conference agenda published ahead of next month's meeting",
    "Central bank releases quarterly economic outlook report",
    "Sports federation announces schedule changes for upcoming season",
    "Social media companies testify before congressional committee today",
    "AI research team publishes findings on neural network efficiency",
    "Environmental agency releases new climate monitoring data",
    "Trade deficit numbers released showing shifts in import export balance",
    "Championship bracket announced for upcoming tournament season",
    "Platform updates terms of service. Users encouraged to review changes",
]

# ── Generate Dataset ──────────────────────────────────────────
rows = []
start_date = datetime(2026, 1, 1)

for i in range(2000):
    topic = random.choice(TOPICS)
    # Realistic sentiment distribution: 40% pos, 35% neg, 25% neutral
    sentiment_roll = random.random()
    if sentiment_roll < 0.40:
        tweet = random.choice(POSITIVE_TWEETS)
        sentiment = 'Positive'
        polarity = round(random.uniform(0.2, 1.0), 4)
        subjectivity = round(random.uniform(0.4, 1.0), 4)
    elif sentiment_roll < 0.75:
        tweet = random.choice(NEGATIVE_TWEETS)
        sentiment = 'Negative'
        polarity = round(random.uniform(-1.0, -0.1), 4)
        subjectivity = round(random.uniform(0.4, 1.0), 4)
    else:
        tweet = random.choice(NEUTRAL_TWEETS)
        sentiment = 'Neutral'
        polarity = round(random.uniform(-0.1, 0.1), 4)
        subjectivity = round(random.uniform(0.0, 0.4), 4)

    # Customize tweet with topic keywords
    date = start_date + timedelta(days=random.randint(0, 117))
    likes = int(np.random.exponential(50))
    retweets = int(np.random.exponential(20))
    replies = int(np.random.exponential(10))

    rows.append({
        'tweet_id': f'TW{str(i+1).zfill(5)}',
        'date': date.strftime('%Y-%m-%d'),
        'tweet': tweet,
        'topic': topic,
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'likes': likes,
        'retweets': retweets,
        'replies': replies,
        'word_count': len(tweet.split()),
        'char_count': len(tweet),
        'has_hashtag': int('#' in tweet),
        'has_emoji': int(any(c in tweet for c in ['🚀','😰','😢','💪','🏆','❤️','🌱','📈','📉','😤','😔','😭','✨','🤖','🌞','🥇','🎉']))
    })

df = pd.DataFrame(rows)
df.to_csv('../data/twitter_sentiment.csv', index=False)
print(f"✅ Dataset generated: {len(df):,} tweets")
print(f"   Sentiment distribution:")
print(df['sentiment'].value_counts())
print(f"   Topics: {df['topic'].nunique()} unique")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
