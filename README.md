# replication-materials-yujing-syj

[![DOI](https://zenodo.org/badge/480985442.svg)](https://zenodo.org/badge/latestdoi/480985442)


Yujing Sun

The code is written in Python 3.8.8 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

## Step 1: Process the Data

- The files used to scrap the data from Reddit are saved in folder `data`. 
- Use the `communities data download.ipynb` file to download all the subreddits from Reddit.  
- For the replication, you don't need to scrape the data again since it takes some time. The data are uploaded here: https://drive.google.com/drive/folders/1z8pn0voDqjsuH3MyAIsEwVf0M3pLPyWe?usp=sharing You can directly download the folder.
- Covid-19 related data is downloaded by https://covid19.who.int/data. The data is in `WHO-COVID-19-global-data.csv`.

## Step 2: Sentiment Analysis and Topic Modeling

### Part 1: Synchronic analysis

- This part contains the following results: 1. Topics of all the posts within WFH community. 2. Sentiment of each topic.
- You can import the `synchronic analysis.py` to generate the `topic sentiment(wfh) raw.csv`. This file is saved in `output` folder, and is used to plot the sentiment of each topic in Tableau. The more detailed results of sentiment analysis and topic modeling are in `synchronic analysis.ipynb`.

### Part 2: Temporal analysis


- This part contains the following results: 1. Monthly sentiment rate, monthly posts number, and Covid-19 new cases. 2. Topic modeling for the posts by three seperate years. 3. Topic modeling for the posts by four different periods when post positive rate is high.
- You can import the `temporal analysis.py` to generate the `reddits sentiment and covid-19.csv`. This file is saved in `output` folder, and is used to plot the relationship between Covid-19 new cases and the sentiment of the posts in Tableau. The more detailed results of sentiment analysis and topic modeling are in `temporal analysis.ipynb`.


- You can import the `WFH sentiment analysis and topic modeling.ipynb` file located in `analysis` folder in this repository to reproduce the analysis.  
- The results of sentiment analysis are saved into `reddits sentiment and covid-19(with graphs).csv` and `topic sentiment(wfh).csv` files. Then I plot the graph based on the results and save the image in the `xlsx`.

## Initial Findings
- Sentiment analysis of subreddits in WFH community from March, 13 2020 to April 14, 2022.  

![image](https://user-images.githubusercontent.com/89925326/165013420-a0e64e67-0bcc-4c7b-b592-6e6341c10cfb.png)

The results show that number of posts in the WFH community is correlated with the number of global Coivd-19 cases, and the correlation is 0.56. For the positive rate of the posts, it is positive correlated (0.45) with the global covid cases. The neutral rate of posts is negative correlated (-0.57) with the severity of pandemic. 

- Sentiment analysis of different topics from WFH community  

  <img src="https://user-images.githubusercontent.com/89925326/165011135-e5570eee-4ef3-4836-90d9-682f3a1b8964.png" width="500" height="300">

After topic modeling, the posts are grouped into several topics, such as job findings, job advertisements, work-life balance, equipment's suggestions (desk, chair, laptop, microphone, software etc), team and coworkers, mental and physical health, back to office and other advice. The overall sentiment of these topics is positive. People are more negative on the topics such as back to office, health issues, time management and work-life balance.
