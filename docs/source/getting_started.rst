===============
Getting started
===============

Installation
------------

The fklearn library is compatible only with Python 3.8+.
In order to install it using pip, run:

.. code-block:: bash

    pip install fklearn

You can also install it from the source:

.. code-block:: bash

    # clone the repository
    git clone -b master https://github.com/nubank/fklearn.git --depth=1
    
    # open the folder
    cd fklearn
    
    # install the dependencies
    pip install -e .


If you are a macOS user, you may need to install some dependencies in order to use LGBM. If you have brew installed,
run the following command from the root dir:

.. code-block:: bash

    brew bundle

Basics
------

Learners
########

While in scikit-learn the main abstraction for a model is a class with the methods ``fit`` and ``transform``,
in fklearn we use what we call a **learner function**. A learner function takes in some training data (plus other parameters),
learns something from it and returns three things: a *prediction function*, the *transformed training data*, and a *log*.

The **prediction function** always has the same signature: it takes in a Pandas dataframe and returns a Pandas dataframe.
It should be able to take in any new dataframe, as long as it contains the required columns, and transform it. The tranform in the fklearn library is equivalent to the transform method of the scikit-learn.
In this case, the prediction function simply creates a new column with the predictions of the linear regression model that was trained.

The **transformed training data** is usually just the prediction function applied to the training data. It is useful when you want predictions on your training set, or for building pipelines, as we’ll see later.

The **log** is a dictionary, and can include any information that is relevant for inspecting or debugging the learner, e.g., what features were used, how many samples there were in the training set, feature importance or coefficients.

Learner functions are usually partially initialized (curried) before being passed to pipelines or applied to data:

.. code-block:: python

    from fklearn.training.regression import linear_regression_learner
    from fklearn.training.transformation import capper, floorer, prediction_ranger

    # initialize several learner functions
    capper_fn = capper(columns_to_cap=["income"], precomputed_caps={"income": 50000})
    regression_fn = linear_regression_learner(features=["income", "bill_amount"], target="spend")
    ranger_fn = prediction_ranger(prediction_min=0.0, prediction_max=20000.0)

    # apply one individually to some data
    p, df, log = regression_fn(training_data)

Available learner functions in fklearn can be found inside the ``fklearn.training`` module.

Pipelines
#########

Learner functions are usually composed into pipelines that apply them in order to data:

.. code-block:: python

    from fklearn.training.pipeline import build_pipeline

    learner = build_pipeline(capper_fn, regression_fn, ranger_fn)
    predict_fn, training_predictions, logs = learner(train_data)

Pipelines behave exactly as individual learner functions. They  guarantee that all steps are applied consistently to both traning and testing/production data.


Validation
##########

Once we have our pipeline defined, we can use fklearn's validation tools to evaluate the performance of our model in different scenarios and using multiple metrics:

.. code-block:: python

    from fklearn.validation.evaluators import r2_evaluator, spearman_evaluator, combined_evaluators
    from fklearn.validation.validator import validator
    from fklearn.validation.splitters import k_fold_splitter, stability_curve_time_splitter

    evaluation_fn = combined_evaluators(evaluators=[r2_evaluator(target_column="spend"),
                                                    spearman_evaluator(target_column="spend")])

    cv_split_fn = k_fold_splitter(n_splits=3, random_state=42)
    stability_split_fn = stability_curve_time_splitter(training_time_limit=pd.to_datetime("2018-01-01"),
                                                       time_column="timestamp")

    cross_validation_results = validator(train_data=train_data,
                                         split_fn=cv_split_fn,
                                         train_fn=learner,
                                         eval_fn=evaluation_fn)

    stability_validation_results = validator(train_data=train_data,
                                             split_fn=stability_split_fn,
                                             train_fn=learner,
                                             eval_fn=evaluation_fn)

The ``validator`` function receives some data, the learner function with our model plus the following:
1. A *splitting function*: these can be found inside the ``fklearn.validation.splitters`` module. They split the data into training and evaluation folds in different ways, simulating situations where training and testing data differ.
2. A *evaluation function*: these can be found inside the ``fklearn.validation.evaluators`` module. They compute various performance metrics of interest on our model's predictions. They can be composed by using ``combined_evaluators`` for example.

Learn More
----------

* Check this `jupyter notebook <https://github.com/nubank/fklearn/blob/master/docs/source/examples/regression.ipynb>`_ for some additional examples.
* Our `blog post <https://medium.com/building-nubank/introducing-fklearn-nubanks-machine-learning-library-part-i-2a1c781035d0>`_ (Part I) gives an overview of the library and motivation behind it.
