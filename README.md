# 💬 Twitter Sentiment Analysis

> End-to-end NLP project analyzing sentiment across **2,000 tweets** covering
> 5 major topics  AI Technology, Climate Change, Economy, Sports & Social Media.

---

## 📊 Executive Dashboard Preview

![Executive Dashboard](visuals/00_executive_dashboard.png)

---

## 📌 Project Overview

Understanding public sentiment on social media is critical for brand monitoring,
crisis detection, and campaign effectiveness measurement.

This project analyzes **2,000 tweets** to answer:
- What is the overall sentiment distribution?
- Which topics generate the most positive/negative sentiment?
- How does sentiment change over time?
- Which sentiment drives the highest engagement?
- What is the relationship between polarity and subjectivity?

---

## 🗂 Project Structure

```
twitter-sentiment-analysis/
├── data/
│   ├── twitter_sentiment.csv           ← Raw dataset (2,000 tweets)
│   └── twitter_sentiment_analyzed.csv  ← Cleaned + analyzed dataset
├── notebooks/
│   └── 01_sentiment_analysis.ipynb     ← Full analysis notebook
├── visuals/
│   ├── 00_executive_dashboard.png      ← Summary dashboard
│   ├── 01_sentiment_distribution.png
│   ├── 02_sentiment_by_topic.png
│   ├── 03_polarity_distribution.png
│   ├── 04_polarity_by_topic.png
│   ├── 05_sentiment_over_time.png
│   ├── 06_engagement_by_sentiment.png
│   └── 07_polarity_vs_subjectivity.png
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

| Finding | Detail |
|---|---|
| **Overall Sentiment** | 41.9% Positive, 32.9% Negative, 25.2% Neutral |
| **Most Positive Topic** | AI Technology  highest avg polarity score |
| **Most Negative Topic** | Social Media  highest negative sentiment |
| **Most Engaging** | Sports tweets drive highest engagement scores |
| **Hashtag Impact** | Tweets with hashtags get 23% more engagement |
| **Subjectivity** | Negative tweets tend to be more subjective |

---

## 💡 Business Recommendations

1. **Brand managers** should monitor Social Media sentiment closely  highest negativity rate
2. **Content teams** should use hashtags and emojis  proven higher engagement
3. **PR teams** should set up real-time alerts for negative sentiment spikes
4. **Marketing teams** can leverage positive AI sentiment for tech product campaigns

---

## 🛠 Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Core analysis |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
| Regex | Text cleaning |
| Jupyter Notebook | Analysis workflow |

---


## 👤 Author

**Haseeb Waqas** — Data Analyst  
📧 haseeb.fr02@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/haseeb-waqas-15531b2a0/)  
