# Databricks notebook source
# DBTITLE 1,Analytical Project_Bank Churners Prediction_M2


# COMMAND ----------

# MAGIC %md
# MAGIC ### Credit card attrition
# MAGIC It is a reduction in credit card users for a company. Most customers of a given business will not remain active customers indefinitely. Whether a one-time purchaser or a loyal customer over many years, every customer will eventually cease his or her relationship with the business. This phenomenon of “disappearing” customers is known by many names, including customer attrition, customer churn, customer turnover, customer cancellation and customer defection.
# MAGIC 
# MAGIC Successfully predicting customer attrition – and proactively preventing it – represents a huge additional potential revenue source for most businesses.
# MAGIC The rate of customer attrition is a key performance indicator (KPI) that businesses need to track in order to make sure they are making the correct strategic decisions. If new customers don’t stick around for long enough, it could be that some customer acquisition expenditures represent a negative ROI for the company (meaning, they lose money for the company).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Models
# MAGIC Logistic Regression, Random Forest and Naive Bayes
# MAGIC #### Inputs (20 Variables)
# MAGIC 1. CLIENTNUM: 
# MAGIC Client number. Unique identifier for the customer holding the account
# MAGIC 
# MAGIC 2. Customer_Age: 
# MAGIC Demographic variable - Customer's Age in Years
# MAGIC 
# MAGIC 3. Gender: 
# MAGIC Demographic variable - M=Male, F=Female
# MAGIC 
# MAGIC 4. Dependent_count: 
# MAGIC Demographic variable - Number of dependents
# MAGIC 
# MAGIC 5. Education_Level: 
# MAGIC Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.)
# MAGIC 
# MAGIC 6. Marital_Status: 
# MAGIC Demographic variable - Married, Single, Divorced, Unknown
# MAGIC 
# MAGIC 7. Income_Category: 
# MAGIC Demographic variable - Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, >)
# MAGIC 
# MAGIC 8. Card_Category: 
# MAGIC Product Variable - Type of Card (Blue, Silver, Gold, Platinum)
# MAGIC 
# MAGIC 9. Months_on_book: 
# MAGIC Period of relationship with bank
# MAGIC 
# MAGIC 10. Total_Relationship_Count
# MAGIC Total no. of products held by the customer
# MAGIC 
# MAGIC 11. Months_Inactive_12_mon
# MAGIC No. of months inactive in the last 12 months
# MAGIC 
# MAGIC 12. Contacts_Count_12_mon
# MAGIC No. of Contacts in the last 12 months
# MAGIC 
# MAGIC 13. Credit_Limit
# MAGIC Credit Limit on the Credit Card
# MAGIC 
# MAGIC 14. Total_Revolving_Bal
# MAGIC Total Revolving Balance on the Credit Card
# MAGIC 
# MAGIC 15. Avg_Open_To_Buy
# MAGIC Open to Buy Credit Line (Average of last 12 months)
# MAGIC 
# MAGIC 16. Total_Amt_Chng_Q4_Q1
# MAGIC Change in Transaction Amount (Q4 over Q1)
# MAGIC 
# MAGIC 17. Total_Trans_Amt
# MAGIC Total Transaction Amount (Last 12 months)
# MAGIC 
# MAGIC 18. Total_Trans_Ct
# MAGIC Total Transaction Count (Last 12 months)
# MAGIC 
# MAGIC 19. Total_Ct_Chng_Q4_Q1
# MAGIC Change in Transaction Count (Q4 over Q1)
# MAGIC 
# MAGIC 20. Avg_Utilization_Ratio
# MAGIC Average Card Utilization Ratio
# MAGIC 
# MAGIC #### Output: "Attrition_Flag"
# MAGIC 
# MAGIC Result Summary: 
# MAGIC 1. The best model is the Random Forest and the worst model is Naive Bayes due to the highest accuracy among the three models.                
# MAGIC 2. Based on the best model_RF, there is a risk of losing 8.9% from existing custmoers.  The reason is that 225 churners are misclassified as existing custmoers which is 8.9% of total 2514 existing customers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data download

# COMMAND ----------

# MAGIC %sh
# MAGIC #pip install mlflow
# MAGIC wget http://mydemoapi.s3.us-east-2.amazonaws.com/BankChurners.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

df = spark.read.csv('file:/databricks/driver/BankChurners.csv', inferSchema=True, header=True)
#df = spark.read.csv('/FileStore/tables/BankChurners.csv', inferSchema=True, header=True, mode='DROPMALFORMED')

# COMMAND ----------

display(df)

# COMMAND ----------

df=df.drop("Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1")
df=df.drop("Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2")
df=df.dropna()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.count()

# COMMAND ----------

display(df.summary())

# COMMAND ----------

# convert intenger column into double 
df = df.withColumn('Customer_Age',df['Customer_Age'].cast("double"))
df = df.withColumn('Dependent_count',df['Dependent_count'].cast("double"))
df = df.withColumn('Months_on_book',df['Months_on_book'].cast("double"))
df = df.withColumn('Total_Relationship_Count',df['Total_Relationship_Count'].cast("double"))
df = df.withColumn('Months_Inactive_12_mon',df['Months_Inactive_12_mon'].cast("double"))
df = df.withColumn('Contacts_Count_12_mon',df['Contacts_Count_12_mon'].cast("double"))
df = df.withColumn('Total_Relationship_Count',df['Total_Relationship_Count'].cast("double"))
df = df.withColumn('Total_Revolving_Bal',df['Total_Revolving_Bal'].cast("double"))
df = df.withColumn('Contacts_Count_12_mon',df['Contacts_Count_12_mon'].cast("double"))
df = df.withColumn('Total_Trans_Amt',df['Total_Trans_Amt'].cast("double"))
df = df.withColumn('Total_Trans_Ct',df['Total_Trans_Ct'].cast("double"))

# COMMAND ----------

df.dtypes

# COMMAND ----------

#data tansformation 
df.select('Education_Level').groupBy('Education_Level').count().show()
df.select('Income_Category').groupBy('Income_Category').count().show()
df.select('Marital_Status').groupBy('Marital_Status').count().show()
df.select('Card_Category').groupBy('Card_Category').count().show()
df.select('Attrition_Flag').groupBy('Attrition_Flag').count().show()


# COMMAND ----------

df.select('Education_Level').groupBy('Education_Level').count().display()

# COMMAND ----------

df.select('Marital_Status').groupBy('Marital_Status').count().display()

# COMMAND ----------

df.select('Attrition_Flag').groupBy('Attrition_Flag').count().display()

# COMMAND ----------

# Review some categorical features 
df.createOrReplaceTempView ('df')

# COMMAND ----------

# MAGIC %sql 
# MAGIC select Income_Category, count(1) as count 
# MAGIC from df 
# MAGIC group by Income_Category 
# MAGIC order by Income_Category desc

# COMMAND ----------

# MAGIC %sql 
# MAGIC select Card_Category, count(1) as count 
# MAGIC from df 
# MAGIC group by Card_Category 
# MAGIC order by Card_Category desc

# COMMAND ----------

# Since 93% of Card_Category are blue, this feature will not be included in the predicting features.

# COMMAND ----------

# Review some numerical features distribution 
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
pdata = df.toPandas()
sns.set(style="darkgrid", font_scale=.9)
fig, ax = plt.subplots()
fig.set_size_inches( 6, 5)
g = sns.distplot(pdata['Credit_Limit'])
display(g)

# COMMAND ----------

fig.set_size_inches( 7, 7)
g1 = sns.displot(pdata['Avg_Utilization_Ratio'])
display(g1)

# COMMAND ----------

# review the features correlation
df_cor = df.toPandas() 
plt.figure(figsize=(20,10))
sns.heatmap(df_cor.corr(), annot=True, linewidth=.5, fmt='.2f', linecolor = 'grey')
plt.show()

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, Model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

# COMMAND ----------

stringIndexer_label = StringIndexer(inputCol="Attrition_Flag", outputCol="label").fit(df)  # product_line -> Double
# Note the above outputCol is label (the predicted column). Here we predict the product line (above) from the attributes below.
#stringIndexer_Att_Flag = StringIndexer(inputCol="Attrition_Flag", outputCol="Attrition_Flag_IX")
stringIndexer_gend = StringIndexer(inputCol="Gender", outputCol="GENDER_IX")
stringIndexer_mar = StringIndexer(inputCol="Marital_Status", outputCol="MARITAL_STATUS_IX")
stringIndexer_edu = StringIndexer(inputCol="Education_Level", outputCol="Education_Level_IX")
# stringIndexer_card = StringIndexer(inputCol="Card_Category", outputCol="Card_Category_IX")
stringIndexer_Income = StringIndexer(inputCol="Income_Category", outputCol="Income_Category_IX")

# COMMAND ----------

#19 predictors
vectorAssembler_features = VectorAssembler(inputCols=["GENDER_IX", "MARITAL_STATUS_IX", "Education_Level_IX","Income_Category_IX","Credit_Limit","Customer_Age","Dependent_count","Months_on_book",
"Months_Inactive_12_mon","Contacts_Count_12_mon","Total_Relationship_Count","Total_Revolving_Bal","Total_Trans_Amt",                 "Total_Trans_Ct","Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio",                        "Total_Relationship_Count"], outputCol="features")

# COMMAND ----------

# Pipeline basic to be shared across model fitting and testing
pipeline = Pipeline(stages=[])  # Must initialize with empty list!

# base pipeline (the processing here should be reused across pipelines)
basePipeline = [stringIndexer_label, stringIndexer_gend, stringIndexer_mar,stringIndexer_edu, 
                stringIndexer_Income,vectorAssembler_features]
# If the above order changes, then the Word Cloud will fail, unless you update the refrence to the CountVectorizers in variable word_model_idx

# COMMAND ----------

#LG, RF, GB, NB models
lr = LogisticRegression()
pl_lr = basePipeline + [lr]
pg_lr = ParamGridBuilder()\
          .baseOn({pipeline.stages: pl_lr})\
          .addGrid(lr.regParam,[0.01, .04])\
          .addGrid(lr.elasticNetParam,[0.1, 0.4])\
          .build()

rf = RandomForestClassifier(numTrees=50)
pl_rf = basePipeline + [rf]
pg_rf = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_rf})\
      .build()

nb = NaiveBayes()
pl_nb = basePipeline + [nb]
pg_nb = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_nb})\
      .addGrid(nb.smoothing,[0.4,1.0])\
      .build()

# One grid from the individual grids
# paramGrid = pg_rf # try fewer models for testing (and to complete sooner)
# Two models:
#paramGrid = pg_lr + pg_rf
# All models:
paramGrid = pg_lr + pg_rf + pg_nb

# COMMAND ----------

#Split the data
splitted_data = df.randomSplit([0.7, 0.3], 24)# proportions [], seed for random
train_data = splitted_data[0]
test_data = splitted_data[1]
#predict_data = splitted_data[2]
print ("Number of training records: " + str(train_data.count()))
print ("Number of testing records : " + str(test_data.count()))
#print ("Number of prediction records : " + str(predict_data.count()))
display(train_data)

# COMMAND ----------

cv = CrossValidator()\
      .setEstimator(pipeline)\
      .setEvaluator(BinaryClassificationEvaluator())\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(2)

#cvModel = cv.fit(training)
cvModel = cv.fit(train_data)
#cvModel = cv.fit(df) # more than 1 hour for all data (if completes)

# COMMAND ----------

#Evaluate models

import numpy as np
# BinaryClassificationEvaluator defaults to ROC AUC, so higher is better
# http://gim.unmc.edu/dxtests/roc3.htm
print("Best Model")
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Worst Model")
print (cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ])

# COMMAND ----------

import re
def paramGrid_model_name(model):
  params = [v for v in model.values() if type(v) is not list]
  name = [v[-1] for v in model.values() if type(v) is list][0]
  name = re.match(r'([a-zA-Z]*)', str(name)).groups()[0]
  return "{}{}".format(name,params)

# Resulting metric and model description
# get the measure from the CrossValidator, cvModel.avgMetrics
# get the model name & params from the paramGrid
# put them together here:
measures = zip(cvModel.avgMetrics, [paramGrid_model_name(m) for m in paramGrid])
metrics,model_names = zip(*measures)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf() # clear figure
fig = plt.figure( figsize=(10, 10))
plt.style.use('fivethirtyeight')
axis = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# plot the metrics as Y
#plt.plot(range(len(model_names)),metrics)
plt.bar(range(len(model_names)),metrics)
# plot the model name & param as X labels
plt.xticks(range(len(model_names)), model_names, rotation=70, fontsize=10)
plt.yticks(fontsize=10)
#plt.xlabel('model',fontsize=8)
plt.ylabel('ROC AUC (better is greater)',fontsize=10)
plt.title('Model evaluations')
display(plt.show())

# COMMAND ----------

predictionsDf = cvModel.transform(test_data)
predictionsDf.registerTempTable('Predictions')
display(predictionsDf)

# COMMAND ----------

predictionsDf.createOrReplaceTempView ('predictionsDf')

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(Gender), Gender
# MAGIC from predictionsDf
# MAGIC where attrition_flag = "Attrited Customer"
# MAGIC group by Gender

# COMMAND ----------

# MAGIC %sql
# MAGIC select customer_age 
# MAGIC from predictionsDf
# MAGIC where attrition_flag = "Attrited Customer"

# COMMAND ----------

numSuccesses = predictionsDf.where("(label = 0 AND prediction = 0) OR  (label = 1 AND prediction = 1)").count()
numInspections = predictionsDf.count()

print ("There were", numInspections, "inspections and there were", numSuccesses, "successful predictions")
print ("This is a", str((float(numSuccesses) / float(numInspections)) * 100) + "%", "success rate")

resultDF = sqlContext.createDataFrame([['correct', numSuccesses], ['incorrect', (numInspections-numSuccesses)]], ['metric', 'value'])
display(resultDF)

# COMMAND ----------

truePositive = int(predictionsDf.where("(label = 1 AND prediction = 1)").count())
trueNegative = int(predictionsDf.where("(label = 0 AND prediction = 0)").count())
falsePositive = int(predictionsDf.where("(label = 0 AND prediction = 1)").count())
falseNegative = int(predictionsDf.where("(label = 1 AND prediction = 0)").count())

print ([['TP', truePositive], ['TN', trueNegative], ['FP', falsePositive], ['FN', falseNegative]])
resultDF = sqlContext.createDataFrame([['TP', truePositive], ['TN', trueNegative], ['FP', falsePositive], ['FN', falseNegative]], ['metric', 'value'])
display(resultDF)
