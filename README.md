<div align="center">
  <b>NBA Rookie Contract Manager</b>
</div>

                                                            
1. ABSTRACT 
                
The 24-25 NBA season is currently on air, and basketball is both  exhilarating and 
unpredictable. For instance, this year the number 1 and number 2 overall pick have been 
underwhelming, yet Jared McCain (number 17) and Dalton Knecht (number 18) have been 
performing very well. The 2023 number 1 pick Victor Wembanyama has been heralded as one of 
the best young talents, and the 2019 number 1 pick Zion Williamson, who was crowned as a future 
superstar destined to win championships, is being slandered for being gluttonous.  
 
Much is unanticipated when it comes to the trajectory of a new basketball player and we can’t rely 
on narratives, only on statistics. Every rookie is signed under a rookie contract, which has two 
guaranteed years, and the third and fourth years are team options - where the team can choose to 
retain them. The player is eligible for contract extension after his third year, although he might not 
get one until his 4th. 
 
Extending a rookie’s contract is a tough decision to make, and it affects the championship status 
of a team by limiting financial flexibility. Our goal is to make this decision completely statistic 
and production based. All rookies will be held to a high bar - the NBA Hall of Fame, and the bar 
will be scaled down to account for role players. If our model is successful, the rookies returned by 
our model should have their contract extended since they are playing well.  
 
2. INTRODUCTION 
 
The agenda is to compare a player’s career averages to those of the Hall of Famers 
(HOFers). These are an elite group of basketball players that any team would like to have. 
However, any good role player or fringe All-Star guy should not be compared to a HOFer, as they 
can still be easily extended by their team but the model would advise not to. Therefore the model 
will be trained on a very high bar, and we will strategically make it more lenient.  
 
The main problem is that the number of HOFers is far lesser than the number of non-HOFers by 
definition. Therefore any dataset we work with will be very imbalanced, yet we cannot 
under/oversample the data to retain positive instances. Moreover, the kind of datasets available in 
the internet often conflict with one another w.r.t missing players, and this might lead to further 
loss of HOF instances.  
 
For this reason, we have to expand the bar from HOF to HOF potential players. This means training 
a classifier to identify HOFers based on the rookie year of their career, then making it predict on 
new data, and making a union of both items to increase HOF instances.  
 
We will then compare new rookies who are still eligible for contract extension, 2017 onwards 
(dataset is up to date till 2021), and let our model predict which ones carry HOF potential. It is at 
this point, we make our model more lenient by threshold tuning - the more lenient the threshold 
is, the more players returned by the model, and the worse kind of players are returned. However, 
all these players should ideally be good enough for their teams to extend.  
 
The only way to fact-check is real-world validation. We will take each returned candidate and 
check whether they were extended/or still playing considerable minutes for another team and 
consider that a success. If not, that is a failure.  
 
Finally, we compute #Successes/#Total candidates as our model accuracy metric. 
 
3. LITERATURE REVIEW 
 
The most amount of work done when it comes to the NBA is related to sports betting, 
more specifically over/under and wins/losses. This is mostly done using neural networks->gradient 
boosting. Data is generally obtained from Basketball Reference or NBA API. These apps are based 
on a Django/Flask backend and generally store postgres data and compare ML models and output 
the best performing model.  
 
Another model was developed on the NBA Elo rating system that was created by Nate Silver. This 
helped because this system simply requires the names of the teams and their elo rating and returns 
win probability, similar to chess.  
 
All the above work is documented in GitHub and the links will be found under the references 
section. On a fundamental level, our problem statements are different. We are trying to evaluate 
individual player performance and its very core, not team performance. Furthermore, the scope of 
the project does not include a software tool, so we do not receive data from an API and do not 
have an interface, although those comprise our scope of improvement (more will be described 
there).  
 
The most similar project used a similar model strategy (SVM->RandomForest->Boosting), to 
calculate All-NBA selections. This makes sense because this model is meant to calculate player 
performance. However, our model works on Hall of Fame potential, which is a different parameter 
altogether. 
 
 
4. MODELS 
 
For the purpose of the HOF potential projection portion of the experiment, I've compared several 
models to choose our best classifier that will serve as a base model.   
 
Performed SMOTE to address imbalance 
 
I compared the following models:  
● Logistic Regression 
● Random Forest 
● SVM 
● Decision Tree 
● KNN 
● Naive Bayes 
 
 
Among these, the best performing model was SVM. To better the performance of SVM, I 
performed the following: 
● Hyperparameter tuning 
● Compared the following betterments: 
1) Boosting (Adaboost) 
2) Bagging 
3) Stacking using Log Reg and Random Forest 
4) Balanced SVM 
5) SMOTE SVM 
6) Majority voting using Log Reg, SVM, and Random Forest ensemble classifier 
● Boosted SVM performs the best, so we use that model to enhance HOFer number from 46 
to 78.  
 
For the purpose of comparing our rookies to these players, performed SMOTE to address 
imbalance, then used Random Forest classifier (this was done to plot feature importance easily). 
 
Finally, to enhance the model, we performed boosting and bagging, where bagging presented 
better results. Used this to predict rookies who should be extended.  
 
 ![image](https://github.com/user-attachments/assets/72a468e9-68d8-495a-87cd-913defeb71d6)

 
Figure 1. High Level Diagram of Project Modelling and Processing 
 
 
5. EXPERIMENT 

5.1. DATASETS  
 
The two datasets we will be using are:  
 
Dataset 1: 
https://www.kaggle.com/datasets/thedevastator/nba-rookies-performance-statistics-and
minutes-p?select=NBA+Rookies+by+Year_Hall+of+Fame+Class.csv  
 
Dataset 2: 
https://www.kaggle.com/datasets/mattop/nba-draft-basketball-player-data-19892021 
             
Dataset 1: (Details the performance statistics of NBA rookies from 1980-2016) 
 
Labels: 1 (HOF) or 0 (non-HOF) (after encoding) 
  
Features:  
Games played, minutes played, points, field goals made, attempted and percentage, 3 
pointers made, attempted and percentage, free throw made, attempted and percentage, 
offensive and defensive rebounds, assists, steals, blocks, turnovers, efficiency ratings. 
 
Dataset 2: (Details the career performance statistic of NBA draft class players from 1989 
to 2021) 
 
Labels: 1 (extended) or 0 (not extended) (after encoding) (This column will added during 
preprocessing)  
 
If accuracy is high in dataset 1, then everyone classified as HOF will be automatically 
classified as extended in Dataset 2. 
 
Features:  
 
(These features include new statistical algorithms that better predict productivity For eg. 
Win shares per 48, box plus minus, Value over replacement player) 
Rank in the draft, Team, College, points, rebounds, assists, field goal percentage, 3 point 
percentage, free throw percentage, avg. minutes, points per game, win shares, win shares 
per 48, box plus minus, value over replacement 
 
 
5.2. PREPROCESSING AND RELATED PROCESSES 
 
1. Dataset 1 has info no later than 2016, therefore all HOFers that have been inducted 
after have to be manually updating by referencing 
https://www.nba.com/news/basketball-hall-of-fame-all-time 
2. If the column ‘Hall of Fame Class’ is filled, we encoded it to 1, otherwise 0. 
Removed any NaN values. 
3. Plotted correlation between features and expanded on NBA terms. Noted 
distribution of said features via histogram 
4. Since we need to increase the number of HOFers, we train on only half the dataset, 
so that we can give us half the dataset to predict on. When we perform union on 
train 1 labels, test 1 labels and predicted 1 labels, we get a higher number of HOF 
potential players. 
5. We return this list of players as a dictionary, and copy it to the second notebook 
where any player in Dataset 2 present in the dictionary is marked 1 in a new column 
called HOF. Players not in the dictionary are marked 0. This new column will serve 
as our label column. 
6. The last HOF potential player is Russell Westbrook, who was drafted in 2008. So 
we divide the dataset into pre 2008, and post 2008. Our model is trained on pre 
2008 data, and post 2008 data will be used for prediction. This also reduces the 
imbalances to half since all post 2008 data with no label 1 is taken out from training 
data. 
7. The parameter importance plot is shown, and features are further analysed 
8. At full strength of the model, the list of future HOFers is returned.  
9. With threshold tuning, the model is made more lenient to return All-Star, All-NBA 
type players.  
10. A subset of rookies who were drafted after 2017 and have played in the NBA for 
less than 4 years is taken. Of these players, the ones which are classified as HOF 
potential are returned in a list. 
11. Finally, real world validation is performed on this data.  
 
5.3. ACCURACY METRICS/ESTIMATES 
                  
Since we are working with standard classifiers, we will be using Precision, F1-score, 
and Recall to determine our model’s performance. Since the data is highly imbalanced,  these 
metrics are preferred over accuracy.  
 
Recall is the most important metric here because recall penalizes false negatives the most. In our 
dataset, false negatives are going to be super problematic, hence recall will be prioritized the most. 

Finally, for real world validation, we will check if returned players were extended by their current 
team or not. Even if they were not extended, they could have been traded and played good minutes 
for other teams. Those instances will also be classified as success. Any other instance is a failure.  
 
 ![image](https://github.com/user-attachments/assets/efacc43f-31e4-4aa7-8fc6-567486d036eb)

 
Figure 2. Formulae of all metrics used to calculate success 
 
 
5.4. RESULTS 
 
Number of Players Returned = 35 
Number of Players Extended/Re-signed = 21 
Number of Players not playing anymore = 4 
Extension prediction rate = 21/35 = 60% success rate 
Good player prediction rate = 31/35 = 88.57% success rate 
 
Let’s visualize using a pie chart,  
 
Green - good results 
Blue - our model performed as it was supposed to but the team that drafted the player did not 
extend for other reasons (budget, roster construction, fit) 
Red - bad results 
7 
 
 ![image](https://github.com/user-attachments/assets/fd1deae1-0e36-4705-a16d-3b86f6fbf9f2)

Figure 3. Model results depicting majority success 
 
 
6. CONCLUSION 
 
 
POSITIVES:  
 
1) The model recognizes talent and good on-court production really well. 
2) Most future HOFers that the model has predicted, it has predicted correctly. 
3) Threshold tuning can give us a range of players. Stricter threshold means better, fewer 
players. More lenient threshold means good players that may not be stars. This can expand 
the problem statement to have the rookie manager keep a tight threshold first, aim to extend 
those players first, and if they still have cap space/money, extend players returned when 
threshold was loosened. 
4) Model requires very little data to categorize a player. For eg, Franz Wagner was drafted in 
2021 and the latest data Dataset 2 has is of 2021. So with just a couple months of Franz’s 
production, it was able to tell that Franz would be extended, which he now has been. 
 
 
DRAWBACKS: 
       
1) The model does not account for many things an on-court scout can. For eg, a player who 
is injury prone but has good production (Eg. Kawhi Leonard, Joel Embiid) will be a definite 
success from the model, but a scout can disagree. If a player on a bad team is taking excess 
shots and making them since the team had poor starters, the model will value that player’s 
empty stats more than another player who sees less minutes since the team already has a 
lot of talent. 
8 
2) The model does not account for current roster construction. If the player is good, the model 
recommends that he should be extended. However, sometimes players can be good but still 
are traded since they might not be a good fit for the current roster, or the team is not willing 
to pay that player the money he deserves because perhaps they are saving it for a star. We 
consider these cases to be successes too, although for the problem statement, they 
technically aren’t.  
3) The HOF potential is drawn from HOFers’ rookie careers. Some HOFers were 
underwhelming in their rookie seasons (because of injuries or less on-court minutes) like 
Steph Curry or Kobe Bryant, but the model does not consider that, and the weights of those 
features might change based on their performance. 
 
Overall, the model does its job well. It’s just that the job requires so many factors that can’t be 
quantified, which could affect the model’s performance.  
 
The user shall only refer to the model to see if any of the returned players are playing for his team. 
The model can guarantee that the player is projected to have a good career. Whether or not to 
extend that player, is a decision that depends on further circumstances.  
 
 
7. SCOPE OF IMPROVEMENT 
 
We mentioned in the literature review that many such projects are continually retrieving data using 
public APIs. The best version of this project is for it to be a product that gets up-to-date data from 
Basketball Reference API / NBA API, etc. It needs to have a frontend that allows the user to adjust 
the threshold using a button/slider and the models are run on the backend.  
 
To account for non-quantifiable data, a confidence score from scouts should be added to the data. 
This can be retrieved from league sources and would help the output be much more nuanced. 
 
From a feature perspective, player compatibility can be tested. If a player is recommended to be 
extended, the user should be able to input the current roster of the team and their production and 
the model should be able to check if the player is a good fit for the team. For eg, if a team is good 
at defense and rebounding, it is perhaps not useful to extend a rookie who is a good defender, but 
is offensively limited. 
 

8. REFERENCES 
  
[1] Roger Sheu, used ML and DL models to determine whether a player will be awarded All-Star 
halfway through the season, or All-NBA at the end of the season. 
https://github.com/rogersheu/All-NBA-Predictions 
 
[2] Kyle Skompinski, created a Flask web app to predict the outcome of NBA games using 
Tensorflow and XGBoost. 
https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting 
 
[3] NBA Betting repository, created an NBA betting dashboard that demonstrates the amount bet 
on NBA games and wins and losses. 
https://github.com/NBA-Betting/NBA_Betting 
 
[4] Luke DiPerna, created a Naive Bayes ML model that predicts the outcome of NBA games 
using box score statistics. 
https://github.com/luke-lite/NBA-Prediction-Modeling 
 
[5] NBA AI repository, created a similar dashboard like NBA betting, using advanced AI processes 
as the data collection and compute cost was becoming too expensive. This streamlined approach 
is based on play by play tracking and visual queue. Because of the huge scope of the project, it is 
still under development. 
https://github.com/NBA-Betting/NBA_AI  
 
 
 
 
 
         
