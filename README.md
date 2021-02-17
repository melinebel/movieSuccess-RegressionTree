# movieSuccess-RegressionTree
A simple regression tree model to study the interplay between box office, critics and audiences in determining a movie's success using data scraped from IMDB.

<h1>Introduction</h1>

<p>What makes a movie successful? While there is a myriad of arguable answers to this question, most people would direct you to the following metrics: Critical acclaim, audience acclaim, and worldwide gross. These three measures describe the mainstream reach and perceived artistic merits of a film and may even feed into each another. So, what does a successful movie – by the above standards - look like? Is it a hundred-million-dollar Hollywood tentpole? Or is it the auteur-driven French film that took home the Palme D’Or? Undoubtedly, there must be an immeasurable number of variables that contribute to a movie’s achievements. Fortunately, there are a few provided by the Internet Movie Database (IMDb) that we can compare against each other to analyze and draw conclusions on the key question of success.</p>
 
<h1>Data</h1>

<p>For this study, I will be using the Kaggle “IMDb movies extensive dataset” scraped from the IMDb website by user Stefano Leone <a href="https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset">(Link)</a>. The “IMDb movies.csv” includes 85,855 movies and the twenty-two following attributes: imdb_title_id, title, original_title, year, date_published, genre, duration, country, language, director, writer, production_company, actors, description, avg_vote, votes, budget, usa_gross_income, worldwide_gross_income, metascore, reviews_from_users, reviews_from_critics.</p>

<h1>Methodology</h1>

<p>Decision trees are versatile supervised algorithms that take in mixed data types and are useful in formulating conclusions about observations in terms of a target variable. Further, the scikit-learn python package offers a DecisionTreeRegressor for modeling with continuous target variables (in our case, wordwide_gross, metascore and avg_rating). For importing, cleaning, describing and preprocessing the dataset, I’ll be using various python packages such as Pandas, NumPy, Matplotlib and Seaborn</p>

<ul>
  <li>After importing the “IMDb movies.csv” dataset I dropped a few attributes that I deemed either irrelevant or problematic for the upcoming model (e.g. 'imdb_title_id', 'title','actors','description','reviews_from_users’ ,'usa_gross_income', etc.). I also cleaned up the remaining characteristics, checking for duplicates and assuring data type congruency.</li>
 
  <li>I did a correlation test between every numerical characteristic. For worldwide_gross_income, the most linearly correlated characteristic is budget (0.74). Metascore and avg_score have the highest correlation to each other (0.73).</li>
  
  <li>The independent and target variables were assigned, and a pipeline was created for the DecisionTreeRegressor() algorithm. This pipeline included calling the OneHotEnconder() function to code the categorical variables of the dataset (genre, country, language). After splitting the dataset and assigning the train and test variables, the GridSearchCV function was called to tune the decision tree parameters and avoid having an overfit model.</li>
  <li>Using the best estimators found by GridSearchCV, I exported the decision tree into a .png image to visualize the model. Both the modeling and visualization was run three times, one for each target variable (worldwide_gross_income, metascore and avg_score).</li>
  </ul>

<h1>Conclusion</h1>

<p>None of the three refined decision tree regressions were particularly effective at explaining the variance of the target variables as measured by r squared. By this standard, the best decision tree is the one that targets the worldwide_gross_income variable, with an r squared of ~0.67. Alternatively, the worst decision tree is the one that targets metascore, with an r squared of ~0.55. These predictive differences make sense in that criticism is a more subjective measurement and it’s likely determined by more abstract, indefinable variables than worldwide income.</p>
Clearly, there is big room for improvement, either by collecting more relevant variables and/or running alternate regression models. Additionally, due to the nature of the OneHotEncoder () methodology, feature names are lost during the modeling pipeline, negatively affecting visualization readability and intelligibility.
