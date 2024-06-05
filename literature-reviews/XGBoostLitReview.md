#### XgBoost vs DNN

Source:
Ali El Bilali, Taleb Abdeslam, Nafii Ayoub, Houda Lamane, Mohamed Abdellah Ezzaouini, Ahmed Elbeltagi,
An interpretable machine learning approach based on DNN, SVR, Extra Tree, and XGBoost models for predicting daily pan evaporation,
Journal of Environmental Management,
Volume 327,
2023,
116890,
ISSN 0301-4797,
https://doi.org/10.1016/j.jenvman.2022.116890.
(https://www.sciencedirect.com/science/article/pii/S030147972202463X)


@article{ELBILALI2023116890,
	author = {Ali {El Bilali} and Taleb Abdeslam and Nafii Ayoub and Houda Lamane and Mohamed Abdellah Ezzaouini and Ahmed Elbeltagi},
	journal = {Journal of Environmental Management},
	pages = {116890},
	title = {An interpretable machine learning approach based on DNN, SVR, Extra Tree, and XGBoost models for predicting daily pan evaporation},
	volume = {327},
	year = {2023}}

Keywords: Interpretable machine learning; Sobol index; SHAP; LIME; Climate variables

Data Source and Preprocessing
- The study focuses on the Bouregreg watershed, specifically the Sidi Mohammed Ben Abdellah (SMBA) dam.
- Climatological data (air temperature, relative humidity, atmospheric pressure, wind speed, and solar radiation) are collected hourly from an automatic weather station.
- Pan evaporation is measured daily.
- Raw data processing involves statistical analysis of hourly climate data to calculate standard deviation, mean, and median values.

Models Used
- The study uses several machine learning models: Extra Tree, XGBoost, Support Vector Regression (SVR), and Deep Neural Network (DNN).
- The models are trained and validated to predict daily pan evaporation.
- Interpretation of the models is conducted using SHAP (for Extra Tree and XGBoost), Sobol indices, and LIME (for DNN and SVR).

Feature Selection Algorithms
- The study employs SHAP and Sobol sensitivity analysis to understand the contribution of input variables to the model outputs.
- This helps in selecting significant features that impact the prediction of pan evaporation.

Evaluation Metrics
- Model performance is evaluated based on their accuracy in predicting daily pan evaporation.
- The consistency of the machine learning models with the evaporative process is also analyzed.
- Model performance metrics included coefficient of correlation (r), RMSE, NSE, MAE, and PBIAS.

Challenges and Limitations
- One challenge is dealing with the variability and complexity of climate data.
- Ensuring the models are interpretable and their predictions are reliable for practical applications.
- The study acknowledges the limitations of small datasets and the need for robust data augmentation methods.

Results
- All models were trained and validated, achieving good accuracy with NSE ranging from 0.8 to 0.86.
- Extra Tree and XGBoost models showed slightly superior performance compared to SVR and DNN.
- Scatter plots and error distributions indicated satisfactory agreement between simulated and observed values of pan evaporation.
- Violin plots demonstrated that model errors were evenly distributed, indicating normality.

The methodology in the evapotranspiration prediction paper employs a variety of machine learning models and emphasizes interpretability using SHAP, Sobol indices, and LIME. Similarly, our models for diagnosing diabetes might also use various algorithms (e.g., logistic regression, neural networks, decision trees). Both methodologies focus on model performance and feature importance but apply to different domains — hydrology vs. healthcare.

While both XGBoost and DNN can be effective for predicting daily pan evaporation, the choice between them depends on the specific requirements of the task, such as the need for interpretability, computational resources, and the nature of the input data. In this particular study, XGBoost demonstrated a slight edge over DNN in terms of prediction accuracy and interpretability.
