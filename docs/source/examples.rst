========
EXAMPLES
========

In this section we will showcase the use of some of Fklearn's functionalities.

Learning Curves
---------------

In Machine Learning applications, as new data gets collected, one might wonder the impact that training a model with
newer data will have in the algorithm's performance. One way to do this is to fix a holdout period and a beginning for the
training, then train the model and evaluate the relevant metric (e.g AUC or MSE). Newer training data should then be
added to the training set and the model should be retrained and reevaluated.

Plotting the performance metric versus the end of the training data yield a *learning curve*. Fklearn has a built-in method
that helps in the process of building learning curves and abstracts much of the process from the hands of the programmer.

Fklearn's `validation` module has a `validator` component that can be used for the building of learning curves. It expects
4 arguments:
  1- The data that will be used for training

  2- A `splitter` function, such as the `time_learning_curve_splitter` from the `validation.splitters` module. This function
  will split the training set according to a predetermined frequency (e.g. 1 month). These are combined to form the training data.
  When running validator, it will initially run the model with the _oldest_ of the splitted data, and successively add
  newer data.

  3- A prediction function. This is the model per se, a function that receives as input a Dataframe and returns, as per
  Fklearn convention, a triple `(predict_fn, predicted_df, log)`, where `predict_fn` is the function that generates the
  prediction column, `predicted_df` is the input Dataframe with the prediction column added and `log` can contain arbitrary
  information.

  4- An evaluation function. This specifies the metric that is being used to evaluate the performance of the model. The
  `validation.evaluators` module contains several predefined functions for computing the most common ML metrics.

Below we present a snippet that stitches together these concepts to produce a function to compute a learning curve for a
simple logistic regression. First, let's define such a model:


.. code-block:: python

  from fklearn.training.classification import logistic_classification_learner

  input_df = get_data_from_source()
  model = logistic_classification_learner(fetures: LIST_OF_MODEL_FEATURES
                                          target: NAME_FOR_TARGET_COL,
                                          prediction_column: NAME_FOR_PREDICTION_COL)


Note that we *did not* pass the data as an input for the function. The reason for this is that Fklearn functions are *curried*,
meaning they receive just a portion of the arguments and return *another function* that has the same behavior as the original,
but receives as arguments the ones that were not passed previously.

.. code-block:: python

  from fklearn.validation import validator, splitters, evaluators

  # define how to split your training data for further evaluation
  learning_curve_split_fn = splitters.time_learning_curve_splitter(training_time_limit=MAX_TRAIN_DATE,
                                                                   time_column=DATE_COL_NAME,
                                                                   freq='M') # split month by month

  # define the metric that should be evaluated. For this example, we use AUC
  eval_fn = evaluators.auc_evaluator(prediction_column=NAME_FOR_PREDICTION_COL,
                                     target_column=NAME_FOR_TARGET_COL)

  # output training logs for the different training ends
  auc_logs = validator.validator(input_df, learning_curve_split_fn, model, eval_fn)


One can also build a *reverse learning curve*, which measures the impact that *old* data has on model performance. In this case,
if a fixed holdout period, one starts training with the most recent data and adds older data in successive steps.

Fklearn also supports the construction of such curves. For such, it is only necessary to change the `splitter function`
passed to the `validator`. This amounts to changing `learning_curve_split_fn` in the previous snippet to:

.. code-block:: python

  learning_curve_split_fn = splitters.reverse_time_learning_curve_splitter(training_time_limit=MAX_TRAIN_DATE,
                                                                           time_column=DATE_COL_NAME,
                                                                           freq='M')
