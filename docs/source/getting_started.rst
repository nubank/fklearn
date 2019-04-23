===============
Getting started
===============

Installation
------------

Fklearn is Python 3.6 compatible only. In order to install it using pip, run::

    pip install fklearn


You can also install from the source::

    git clone git@github.com:nubank/fklearn.git
    cd fklearn
    git checkout master
    pip install -e .


If you are a MacOs user, you may need to install some dependencies in order to use LGBM. If you have brew installed,
run the following command from the root dir::

    brew bundle

Basics
------

Learners
########

While in scikit-learn the main abstraction for a model is a class with methods ``fit`` and ``transform``,
in fklearn we use what we call a **learner function**. A learner function takes in some training data (plus other parameters),
learns something from it and returns three things: a *prediction function*, the *transformed training data*, and a *log*.
As an example, here’s a simplified definition of the ``linear_regression_learner``::

    from sklearn.linear_model import LinearRegression
    from toolz import curry

    @curry
    def linear_regression_learner(df: pd.DataFrame,
                                                       features: List[str],
                                                       target: str,
                                                       params: Dict[str, Any] = None) -> LearnerReturnType:

       # initialize and fit the linear regression
       reg = LinearRegression(**params)
       reg.fit(df[features].values, df[target].values)

       # define the prediction function
       def p(new_df: pd.DataFrame) -> pd.DataFrame:
           # note that `reg` here refers to the linear regression fit above, via the function’s closure.
           return new_df.assign(prediction=reg.predict(new_df[features].values))

       # the log can contain arbitrary information that helps inspect or debug the model
       log = {'linear_regression_learner': {
           'features': features,
           'target': target,
           'parameters': params,
           'training_samples': len(df),
           'feature_importance': dict(zip(features, reg.coef_.flatten()))
       }

       return p, p(df), log

Notice the use of type hints! They help make functional programming in python less awkward, along with the immensely useful `toolz <https://toolz.readthedocs.io>`_ library.

As we mentioned, a *learner function* returns three things (a function, a dataframe, and a dictionary), as described by the ``LearnerReturnType`` definition::

    LearnerReturnType = Tuple[PredictFnType, pd.DataFrame, LearnerLogType]
    PredictFnType = Callable[[pd.DataFrame], pd.DataFrame]
    LearnerLogType = Dict[str, Any]

The **prediction function** always has the same signature: it takes in a dataframe and returns a dataframe (we use Pandas dataframes).
It should be able to take in any new dataframe (as long as it contains the required columns) and transform it
(it is equivalent to the transform method of a scikit-learn object).
In this case, the prediction function simply creates a new column with the predictions of the linear regression model that was trained.

The **transformed training data** is usually just the prediction function applied to the training data. It is useful when you want predictions on your training set, or for building pipelines, as we’ll see later.

The **log** is a dictionary, and can include any information that is relevant for inspecting or debugging the learner (e.g. what features were used, how many samples there were in the training set, feature importance or coefficients).

*Learner functions* show some common functional programming properties:
They are **pure functions**, meaning they always return the same result given the same input, and they have no side-effects. In practice, this means you can call the learner as many times as you want without worrying about getting inconsistent results. This is not always the case when calling fit on a scikit-learn object for example, as objects may mutate.

They are **higher order functions**, as they return another function (the prediction function). As the prediction function is defined within the learner itself, it can access variables in the learner function’s scope via its closure.

By having consistent signatures, learner functions (and prediction functions) are **composable**. This means building entire pipelines out of them is straightforward, as we’ll see soon.

They are **curriable**, meaning you can initialize them in steps, passing just a few arguments at a time (this is what’s actually happening in the first three lines of our example). This will be useful when defining pipelines, and applying a single model to different datasets while getting consistent results.


Pipelines
#########



Validation
##########

Learn More
----------

