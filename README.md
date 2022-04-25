# replication-materials-yujing-syj

Yujing Sun

The code is written in Python 3.8.8 and all of its dependencies can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository):

```
pip install -r requirements.txt
```

## Step 1: Process the Data

- The files used to scrap the data from Reddit are saved in folder `data`. 
- Use the `communities data download.ipynb` file to download all the subreddits from Reddit.  
- The data are uploaded here: https://drive.google.com/drive/folders/1z8pn0voDqjsuH3MyAIsEwVf0M3pLPyWe?usp=sharing 
- Covid-19 related data is downloaded by https://covid19.who.int/data. The data is in `WHO-COVID-19-global-data.csv`.

## Step 2: Sentiment Analysis and Topic Modeling

- You can import the `WFH sentiment analysis and topic modeling.ipynb` file located in `analysis` folder in this repository to reproduce the analysis.  
- The results of sentiment analysis are saved into `reddits sentiment and covid-19(with graphs).csv` and `topic sentiment(wfh).csv` files. Then I plot the graph based on the results and save the image in the `excel`.

## Initial Findings
- Sentiment analysis of subreddits in WFH community from March, 13 2020 to April 14, 2022.  

![image](https://user-images.githubusercontent.com/89925326/165013420-a0e64e67-0bcc-4c7b-b592-6e6341c10cfb.png)

The results show that number of posts in the WFH community is correlated with the number of global Coivd-19 cases, and the correlation is 0.56. For the positive rate of the posts, they are positive correlated (0.45) with the global covid cases. While the neutral rate of posts are negative correlated (-0.57) with the severity of pandemic. 

- Sentiment analysis of different topics from WFH community  

  <img src="https://user-images.githubusercontent.com/89925326/165011135-e5570eee-4ef3-4836-90d9-682f3a1b8964.png" width="500" height="300">

After topic modeling, the posts are grouped into several topics, such as job findings, job advertisements, work-life balance, equipment's suggestions (desk, chair, laptop, microphone, software etc), team and coworkers, mental and physical health, back to office and other advice. The overall sentiment of these topic is positive. People are more negative on the topics such as back to office, health issues, time management and work-life balance.
