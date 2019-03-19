
def learner_pred_fn_docstring(f_name: str, shap: bool = False) -> str:
    shap_docstring = """
    apply_shap : boolean, optional
        Creates a new output column named "shap" with SHAP values.
        SHAP values can be used for interpretability and feature importances.
        For more information, see https://github.com/slundberg/shap

    """ if shap else ""

    docstring = """
    Predict function from %s

    Parameters
    ----------
    new_df : pandas.DataFrame
        A Pandas' DataFrame with the same columns as the one
        used to train the learner.
    %s
    Returns
    -------
    df : pandas.DataFrame
        A `new_df`-like DataFrame with the same columns as the
        input `new_df` plus a column with predictions from the trained learner.

    """ % (f_name, shap_docstring)

    return docstring


def learner_return_docstring(model_name: str) -> str:
    docstring = """
    Returns
    ----------
    p : function pandas.DataFrame -> pandas.DataFrame
        A function that when applied to a DataFrame with the same columns as `df`
        returns a new DataFrame with a new column with predictions from the model.

    new_df : pandas.DataFrame
        A `df`-like DataFrame with the same columns as the input `df` plus a
        column with predictions from the model.

    log : dict
        A log-like Dict that stores information of the %s model.""" % model_name

    return docstring


splitter_return_docstring = """

    Returns
    ----------
    Folds : list of tuples
        A list of folds. Each fold is a Tuple of arrays.
        The fist array in each tuple contains training indexes while the second
        array contains validation indexes.

    logs : list of dict
        A list of logs, one for each fold"""
