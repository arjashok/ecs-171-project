# ECS 171 -- Term Project #
Group 6: Arjun Ashok (<arjashok@ucdavis.edu>), Tej Sidhu (<tejsidhu@ucdavis.edu>), 
Taha Abdullah (<tmabdullah@ucdavis.edu>), Ayush Tripathi (<atripathi@ucdavis.edu>), 
Devon Streelman (<djstreelman@ucdavis.edu>)

# Problem Statement

Diabetes affects nearly 38.5 million people in the United States, a
staggering 11.6% of the population. As common as diabetes is, the
disease has serious complications, including nerve damage, heart
disease, chronic kidney disease, and other complications such as oral,
vision, and mental health. Despite the prevalence of this disease, it
can go unnoticed. This widespread epidemic has warning signs that can be
utilized to predict the disease\'s presence, or allow people to catch
the presence of the disease early. Our algorithm will be able to analyze
these features and give us a model to accurately predict diabetes in a
patient who unknowingly has the condition.
[(according to the CDC)](https://www.cdc.gov/diabetes/index.html)

# Dataset

Dataset:
[diabetes-health-indicators-dataset @ Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), 
[diabetes.csv](https://github.com/ArjunAshok17/ecs-171-project/blob/main/datasets/diabetes.csv)

To accurately predict diabetes risk, we needed a comprehensive dataset
with a lot of observations and features. Luckily, we were able to find a
dataset on Kaggle which had around 250,000 observations and 22 features.
This was the perfect dataset for this project because we had enough
features to do feature selection and enough observations to thoroughly
train our model and ensure that it makes accurate predictions.
Additionally, the dataset had some outliers and correlations which meant
that we had enough noise and variation in the dataset to effectively
incorporate data cleaning and exploratory data analysis into the
project.

# Goal

Our goal with this project is to leverage the diabetes survey data for
(a) an improved, *and fair* (i.e. across income and sex), prediction of
diabetes risk given a set of patient characteristics, (b) detailed
analysis of behaviors or qualities that patients possess that can
increase/decrease the risk of diabetes, and (c) novel methods for
dealing with survey data during the data cleaning/transformation and
modeling phases to improve performance across related predictive
contexts. To accomplish this, we'll develop package level code
(performant, clean, modular, and plenty of abstractions) that can be
leveraged by others in the hopes of easy generalization into other
contexts.

# Deliverables

|**Deliverable**                                            | **Due Date**|
|-----------------------------------------------------------|-----------|
|EDA + Feature Selection                                    | 5/05|
|Model Construction                                         | 5/19|
|Model Evaluation / Fine-Tuning                             | 5/26|
|Model Deployment / Fine-Tuning                             | 6/02|
|Application / Final Paper Submission                       | 6/08|
