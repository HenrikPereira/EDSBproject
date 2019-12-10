<p>
    <img src = "https://www.novaims.unl.pt/images/logo.png" width = 100>
    <b><span style="font-size: x-large; ">Human Resources Analysis - Predicting Attrition</span></b>
</p>
<h2 align=left><span style="font-size: medium; ">EDS Bootcamp project | Enterprise Data Science and Analytics (2019)</span></h2>

## Use Case: Human Resources Analysis Predicting Attrition
*Authors: Bruno Candeias<sup>1</sup>, David Oliveira<sup>2</sup>, Henrique Pereira<sup>3</sup> & Manuel Oom<sup>4</sup>*
1. *M20180313: M20180313@novaims.unl.pt*
2. *M20181430: M20181430@novaims.unl.pt*
3. *M20181395: M20181395@novaims.unl.pt*
4. *M20181431: M20181431@novaims.unl.pt*

### I. Important Files
The most important files in this work are the following:
* Final Presentation_EDSB_20191209_v0.01.ppt
* HumanResourcesAnalysis_PredictAttrition.pbix
* Data_models.ipynb
* data_pre_proc.py
* auxiliary.py

Please check the requirements file for further information.

### II. Status Report
We choose this use case mainly because it allows us to explore different models, which will give us the opportunity to 
have a broader view for the problem: descriptive and predictive. In addition, human resources turnover is very present 
in our professional life, which also motivated our choice.

Our approach will follow the work flow: 
1. Data Exploration; 
2. Model Evaluation & Selection; 
3. Results Presentation. 

### III. Dataset Exploration
The dataset (HR_DS.csv) that will be used in the use case (Human Resources Analysis Predict Attrition) contains 1470 
records with 35 columns:
* Age;	Attrition;	BusinessTravel;	DailyRate; Department; DistanceFromHome; Education; EducationField; EmployeeCount;
 EmployeeNumber; EnvironmentSatisfaction; Gender; HourlyRate; JobInvolvement; JobLevel; JobRole; JobSatisfaction; 
 MaritalStatus; MonthlyIncome; MonthlyRate; NumCompaniesWorked; Over18; OverTime; PercentSalaryHike; PerformanceRating; 
 RelationshipSatisfaction; StandardHours; StockOptionLevel; TotalWorkingYears; TrainingTimesLastYear; WorkLifeBalance; 
 YearsAtCompany; YearsInCurrentRole; YearsSinceLastPromotion; YearsWithCurrManager 

We'll explore the dataset to evaluate each variable and how they are correlated. Our first findings were: 
* Most of our data (DistanceFromHome, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, TotalWorkingYear, 
YearsAtCompany, YearsSinceLastPromotion) shows skewness, and not normal;
* Columns like YearsWithCurrManager and YearsInCurrentRole have 2 different distributions with a cutoff by 5 years;
* There are several variables that have outliers: MonthlyIncome, NumCompaniesWorked, PerformanceRating, StockOptions, 
TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, 
YearsWithCurrManager;
* Regarding the variable Attrition and how it is correlated with other variables, we analyzed the data and we realized
that younger employees leave more in all categories, except SalesExecutive, ManufacturingDirector, Manager and 
Divorced. We also realized that the gender is important in some conditions.

Regarding the block Model Evaluation & Selection, we will study several predictive and classification models to 
apply in the dataset related with the use case, namely:
* XGBoost;
* Logistic Regression Classifier;
* Linear Support Vector Classification;
* C-Support Vector Classification;
* Random Forest Classifier;
* Keras Deep Neural Network.
