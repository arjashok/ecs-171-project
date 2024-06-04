### Literature Review


Source: 
J. P. Li, A. U. Haq, S. U. Din, J. Khan, A. Khan and A. Saboor, "Heart Disease Identification Method Using Machine Learning Classification in E-Healthcare," in IEEE Access, vol. 8, pp. 107562-107582, 2020, doi: 10.1109/ACCESS.2020.3001149. keywords: {Feature extraction;Diseases;Machine learning;Heart;Prediction algorithms;Machine learning algorithms;Support vector machines;Heart disease classification;features selection;disease diagnosis;intelligent system;medical data analytics},
https://ieeexplore.ieee.org/document/9112202

Data source and preprocessing
- Cleveland Heart Disease dataset
- Removed attribute missing values
- SS (Standard Scalar)
- Min-Max Scalar 

Models Used
- LR (Linear Regression) 
- KNN (Kth Nearest Neighbor)
- ANN (Artificial Neural Networks)
- SVM (Support Vector Machine)
- NB (Naive Bayes)
- DT (Decision Tree)

Feature Selection Algorithms
- Relief
- MRMR (Maximum Relevance - Minimum Redundancy)
- LASSO
- LLBFS (Layer-Wise Learning Based Feature Selection)
- FCMIM (Fast conditional Mutual Information Maximization)

Evaluation Metrics
- Accuracy
- Sensitivity
- Specificity
- MCC
- LOSO
- Calculated with the help of a confusion matrix

Challenges and limitations?
- No obvious challenges


In literature we found that diagnosis through machine learning has taken similar approaches to ours in their search to diagnose heart disease. Li et al created their models using the Cleveland Heart Disease dataset. Throughout their preprocessing, the researchers removed attribute missing values and scaled their data with both a standard scalar technique as well as the min-max scalar technique. The researchers applied various classification models to the data throughout their testing, including linear regression(LR), Kth nearest neighbor (KNN), artificial neural network (ANN), support vector machine (SVM), naive bayes (NB), and decision trees (DT). There was also a variety of feature selection algorithms used, including Relief, MRMR (Maximum Relevance - Minimum Redundancy), LASSO and LLBFS LLBFS (Layer-Wise Learning Based Feature Selection). After extensive testing of each classification model with the respective feature selective  algorithm, the proposed algorithm was FCMIM (Fast conditional Mutual Information Maximization) because of its performance in comparison to the other feature selection algorithms. The researchers reported the accuracy of the models using FCMIM, with LR at 88.67%, KNN at 82.11%, ANN at 75.23%, SVM at 92.37%, NB at 86.01%, and DT at 79.12%. The model and feature selection with the best results was FCMIM-SVM with 92% accuracy.