# ml-predict-infant-mortality


Infant mortality is an important indication of the overall health situation of a nation. This study evaluates the performances of popular machine learning models such as  Decision Trees (DTs), Random Forests (RFs), Support Vector Machines (SVM), and  Logistic Regression (LR) to predict infant mortality in Bangladesh. We consider  Bangladesh Demographic and Health Survey (BDHS) 2014 data to predict infant  mortality. Therefore, feature selection techniques become an inevitable part of our  analysis. Three more accepted feature selection techniques, for instance, Random  Forests based algorithm, Boruta algorithm, and SVM with the linear kernel-based  algorithm have used in this analysis. Hence, the selected features are Husband/partner's  occupation (V704), Body Mass Index (BMI), Type of cooking fuel (V161), Total  children ever born (V201), Birth order number (BORD), Mother Age at 1st birth  (V212), Division (V024), Wealth index (V190) and Husband/partner's education level  (V701) for classifying infant mortality in Bangladesh. Hereafter, machine learning  models' performance is evaluated using confusion matrix, ROC curve, and k-fold cross-validation. Results illustrated that SVM with Gaussian kernel performs better to predict infant mortality in Bangladesh. 


Justification of the study


The  first year of an infant’s life is of special importance to ensure its health substructure (Sharifzadeh, 2008). Infant and child mortality rates reflect a country’s level of socioeconomic development and therefore, used for monitoring and evaluating population and health programs and policies. (BDHS 2014). So, if we can predict the risk factors of infant mortality, it can be helpful for the policymaker to construct a plan to flourish the health situation of the nation.



Objective of the study

Evaluating the performance of different Machine Learning algorithms using confusion matrix, ROC curve, and k-fold cross-validation

Implementation of Machine Learning models, such as decision tree, random forest, support vector machine, and logistic regression for classifying the infant mortality in Bangladesh

Classification of multiple categories with a better machine learning model comparing other models based on the results.





Study Design


EDA on Infant Mortality:






Figure : Scatter diagram of the infants who died before the first birthday in correspondence to BMI & Mothers age at birth


The figure is the scatter diagram of the infants who died before their first birthday in correspondence to Body Mass Index & Mothers age at birth. It shows that most of the children died before 1st birthday if the body mass index is on the higher side. If the BMI is on the lower side the percentage of the child who died is much less. It also shows that Infant Mortality has also an association with mother age at birth. If the mother gave birth at an early age, there is more possibility of the death of an infant. If the mother gave birth at delayed age then in most cases the infant survived.





Figure : Pair plots of different variables in correspondence with infant mortality


Relative Importance of Predictors:





Figure : Features selection using Random Forests.


Decision Trees and Random Forests do not require any assumption, but SVM and Logistic regression require the assumption of independence among predictors variables. Hence, we are exploring important features using SVM. Once having the fitted SVM with a linear kernel, then the important features can be determined by comparing the size of the classifier coefficients using .coef_. The figure presents the main features used in classification with the blue bars and the not important ones (which hold less variance) with the green bars.





Figure : Features selection using SVM.


Evaluation of Machine Learning Models


Before using any machine learning model, for instance, Decision Trees (DT), Random Forest (RF), Support Vector Machine (SVM), Logistic Regression, etc., we need to identify the important variables to predict our focused characteristics. Hence, the relative importance of predictors is required in the analysis. 

Boruta algorithm was performed to extract relevant risk factors for first-day mortality from the BDHS-2014 dataset. This is a wrapper build algorithm around the random forest classifier to find out the relevance by comparing features to the random probes (Kursa and Rudnicki, 2010). Hereafter, the main features were identified using Random Forests and then SVM. With the aid of the SVM algorithm, nine variables (V704, BMI, V161, , V201, BORD, V212, V024, V190, and V701)  were selected for classifying infant mortality. Hereafter, these 8 variables are used in this section to evaluate the performance of the machine learning algorithm. The analysis of machine learning algorithms was completed using the scikit-learn module in Python version 3.7.3. We considered the confusion matrix, the receiver operating characteristic (ROC), and the k-fold cross-validation approaches to evaluate the performance. We used, 1000 decision trees and Ginni for impurity index to implement the random forests algorithm in Python.


Confusion Matrices of Machine Learning Models






These figures illustrates different confusion matrices of decision trees (DTs), random forests (RFs), support vector machine with rbf kernel, and logistic regression. The confusion matrices were calculated using the scikit-learn module considering 70% observations as training data and 30% observation as test data with random seed 3299 in Python version 3.7.3. Table 5 reveals accuracy scores, sensitivity, specificity, and precisions of all mentioned machine learning algorithms. 


Table : Accuracy, Sensitivity, Specificity, and Precisions of Machine Learning Models.

Models

Accuracy

Sensitivity

Specificity

Precision

Decision Trees

0.802

0.882

0.315

0.886

Random Forests

0.836

0.869

0.352

0.953

SVM (with rbf kernel )

0.840

0.861

0.310

0.970

Logistic Regression

0.854

0.854

-

1.000


The accuracy, sensitivity, specificity, and precision value of respective machine learning models are showed in this table. With an accuracy score of 0.854, the logistic regression gives the best accuracy among the discussed machine learning models. Nevertheless, the logistic regression could not be calculated due to the convergence problem. Thus the support vector machine with rbf kernel performs well among these models regarding this condition.


ROC of Machine Learning Models


The ROCs were calculated using the scikit-learn module with random seed 100 in Python version 3.7.3, considering 70% observations as training data and 30% observation as test data. The area under the ROC curve (AUC) was estimated and plotted in Figure 3.32 to 3.35 The highest AUC was observed for Random Forests (0.68) followed by SVM with Gaussian Kernel (0.62), Decision Trees (0.60), and Logistic Regression (0.53).







K-Fold Cross-Validation of Machine Learning Models


The following represents that the support vector machine with rbf kernel performed better in 5-Fold, 10-Fold, and 30-Fold cross-validation. So, to predict infant mortality the support vector machine with rbf kernel performs better.



Conclusion


The prediction of infant mortality has been made possible by evaluating the machine  learning models. To evaluate the machine learning models, different steps of analysis  has been performed to make a meaningful conclusion.  Firstly the exploratory data analysis (EDA) is being used to get profound information  about the data which considers only the birth record file (bdbr70sv) from the  Bangladesh Demographic and Health Survey (BDHS 2014). Performance of these  machine learning models e.g., decision trees (DTs), random forests (RFs), support  vector machines (SVMs), logistic regression is evaluated by confusion matrix, the  receiver operating characteristic (ROC), and the k-fold cross-validation. After the exploratory data analysis the variables' significance is measured by the chi-square test and Eta statistics. We also used three renowned feature selection algorithms,  e.g., Random Forests based algorithm, Boruta algorithm, and SVM with the linear  kernel-based algorithm. Hence, we found nine significant variables, which are  Husband/partner's occupation (V704), Body Mass Index (BMI), Type of cooking fuel  (V161), Total children ever born (V201), Birth order number (BORD), Mother Age at  1st birth (V212), Division (V024), Wealth index (V190) and Husband/partner's  education level (V701). At the beginning of evaluating the machine learning models, we use the confusion  matrix and get the accuracy, sensitivity, specificity, and precision scores. The support  vector machine with rbf kernel has got an accuracy score of 0.854. Although logistic  regression is omitted due to the convergence problem. The ROCs were calculated using  the scikit-learn module with random seed 100 in Python version 3.7.3, considering 70%  observations as training data and 30% observation as test data. The area under the ROC curve (AUC) was estimated and plotted in Figure 3.32 to 3.35 The highest AUC was  observed for Random Forests (0.68) followed by SVM with Gaussian Kernel (0.62),  Decision Trees (0.60), and Logistic Regression (0.53). At last, in the k-fold cross- validation of the machine learning models, the support vector machine with rbf kernel  performs better among the machine learning models with an accuracy of 0.808, 0.807,  0.808 in 5-Fold, 10-Fold, and 30-Fold cross-validation. Hence the SVM with Gaussian  (rbf) kernel is the best classifier to predict infant mortality in Bangladesh using BDHS  2014 data
