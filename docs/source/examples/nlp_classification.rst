:orphan:

NLP Classification
==================

In this example we will use
the `Consumer Complaint Database <https://catalog.data.gov/dataset/consumer-complaint-database>`_
to predict whether the consumer is asking a question about "Credit reporting, credit repair services,
or other personal consumer reports", using the question as a feature.

Since this is a binary classification task we will use TF-IDF and Logistic Regression, the
baseline of any NLP classification task, as our model (available as
``nlp_logistic_classification_learner`` on fklearn).

.. code-block:: python

   import pandas as pd
   from sklearn.metrics import accuracy_score
   from fklearn.preprocessing.splitting import time_split_dataset
   from fklearn.training.classification import nlp_logistic_classification_learner
   from fklearn.validation.evaluators import fbeta_score_evaluator


   # Load consumer complaints data
   def load_data(path):
      df = pd.read_csv(path, usecols=["Product", "Consumer complaint narrative", "Date received", "Complaint ID"], parse_dates=["Date received"])\
             .rename(columns={"Product": "product", "Consumer complaint narrative": "text", "Date received": "time", "Complaint ID": "id"})
      df["target"] = (df["product"] == "Credit reporting, credit repair services, or other personal consumer reports").astype(int)
      return df.dropna()

   df = load_data("Consumer_Complaints.csv")

   # Split using the `time` column, using 2017 to train and 2018 to test
   train, holdout = time_split_dataset(df, train_start_date="2017-01-01", train_end_date="2018-01-01", holdout_end_date="2019-01-01", time_column="time")

   # Train model and predict on holdout data
   predict_fn, train_pred, logs = nlp_logistic_classification_learner(train, text_feature_cols=["text"], target="target")
   holdout_pred = predict_fn(holdout)


   # Measure F1-Score
   f1_score = fbeta_score_evaluator(holdout_pred)
   # {'fbeta_evaluator__target': 0.7611906547172731}
