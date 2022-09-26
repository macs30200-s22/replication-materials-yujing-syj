# Working remotely as a new norm: How have people's attitudes toward work from home on Reddit changed over the course of the pandemic?

[![DOI](https://zenodo.org/badge/480985442.svg)](https://zenodo.org/badge/latestdoi/480985442)


Yujing Sun

The code is written in Python 3.8.8 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

## Step 1: Process the Data

- The files used to scrap the data from Reddit (`build_user_subreddit_history.py` and `download_subreddits.py`) are saved in folder `data`. 
- Use the `communities data download.ipynb` file to download all the subreddits from Reddit.  
- For the replication, you don't need to scrape the data again since it takes some time. The data is uploaded here: https://drive.google.com/drive/folders/1z8pn0voDqjsuH3MyAIsEwVf0M3pLPyWe?usp=sharing. You can directly download the folder.
- Covid-19 related data is downloaded by https://covid19.who.int/data. The data is in `WHO-COVID-19-global-data.csv`. I filter the raw data to get monthly new cases data `Covid-19 new cases.csv` that I will use in my analysis part.

## Step 2: Sentiment Analysis and Topic Modeling

### Part 0: Replication preparation

- In order to rerun all the results smoothly, you should put `requirements.txt`, `new_data` folder (which could be downloaded by the google drive link), `Covid-19 new cases.csv`, `synchronic analysis.py`, `synchronic analysis.ipynb`, `temporal analysis.py`, and `temporal analysis.ipynb` into same path.

### Part 1: Synchronic analysis

- This part contains the following results: 
  -  Topics of all the posts within WFH community. 
  -  Sentiment of each topic.
- You can import the `synchronic analysis.py` located in `analysis` folder to generate the `topic sentiment(wfh) raw.csv`. This file is saved in `output` folder, and is used to plot the sentiment of each topic in Tableau. The more detailed results of sentiment analysis and topic modeling are in `synchronic analysis.ipynb`.

### Part 2: Temporal analysis

- This part contains the following results: 
  -  Monthly sentiment rate, monthly posts number, and Covid-19 new cases. 
  -  Topic modeling for the posts by three seperate years. 
  -  Topic modeling for the posts by four different periods when post positive rate is high.
- You can import the `temporal analysis.py` located in `analysis` folder to generate the `reddits sentiment and covid-19.csv`. This file is saved in `output` folder, and is used to plot the relationship between Covid-19 new cases and the sentiment of the posts in Tableau. The more detailed results of sentiment analysis and topic modeling are in `temporal analysis.ipynb`.



## Findings

- I plot all the graphs by Tableau. The images are saved in `outputs` folder.


   
### Part 1: Synchronic analysis
   
- Topics within r/workfromhome community
    
<img src="./outputs/top 11 topics.png" width="500" height="300">
     
- Sentiment analysis of subreddits in WFH community from March, 13 2020 to April 14, 2022.  
    
<img src="./outputs/sentiment analysis of different topics.png" width="600" height="300">
     
From the result, “recommendation”, “survey”, “equipment”, “make money online” and “advertisements” are relatively positive compared to other posts. In contrast, “back to office”, “physical and mental health”, “team and coworkers”, “life style and work-life balance” and “time management” are relatively negative topics.    
   
### Part 2: Temporal analysis
    
- Relationship between post's number and Covid-19 new cases
     
<img src="./outputs/trend of monthly posts number and Covid-19 cases.png" width="600" height="300">
       
The number of posts follow some similar pattern with the number of Covid-19 new cases in the world. The correlation between Global Covid-19 new cases and the volume of posts is 0.41 and the correlation between percentage change of Global Covid-19 new cases and percentage change of the volume of posts is 0.51.
    
- Relationship between posts sentiment and Covid-19  
    
<img src="./outputs/trend of monthly posts positive rate and Covid-19 cases.png" width="600" height="400">
     
The positive rate is high at the beginning of the pandemic, then the rate falls until the first wave of Covid-19. The posts positive rate is relatively high in the first, second and fourth waves of Covid-19. 
     
- Topics for different years
    
<img src="./outputs/topic for each year.png" width="600" height="500">
      
The change of topics reflects the phenomenon that at first, people like to discuss some urgent issue in terms of working from home. As people have some experience of the new working style, their focus is more diverse and personal. 
      
   
