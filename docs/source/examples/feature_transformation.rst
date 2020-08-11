:orphan:

Feature transformations
===============

Feature engineering is an important part of every Machine Learning project, either because you need to convert your categorical features to numerical values, or because the numerical features need some special transformation.

Fklearn`s ``transformation`` module contains a number of different feature transformations that can be incorporated in your feature engineering.

Available Encoders
^^^^^

Categorical:
- Replace rare categories with a single one
- Rank by frequency encoder
- Frequency encoder
- Label encoder
- Onehot
- Target encoder

Numerical:
- Capper
- Floorer
- Quantile Biner
- Standard Scaler


Usage
^^^^^

Below we present an example of encoder usage, applying a ``target_categorizer`` to a dataset and either replacing or storing the original values.


.. code-block:: python

    from fklearn.training.transformation import target_categorizer
    import pandas as pd

    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])

    # Replace
    pipe = target_categorizer(columns_to_categorize=['b', 'a'], 
                              target_column='c')
    p, p_df0, log = pipe(df)

    # Store originals in columns listed in a dict
    pipe = target_categorizer(columns_to_categorize=['b', 'a'], 
                              target_column='c', columns_mapping={'b': 'b_raw'})
    p, p_df1, log = pipe(df)

    # Add prefix to the columns with original values
    pipe = target_categorizer(columns_to_categorize=['b', 'a'], 
                              target_column='c', prefix='raw__')
    p, p_df2, log = pipe(df)

    # Add suffix to the columns with original values
    pipe = target_categorizer(columns_to_categorize=['b', 'a'], 
                              target_column='c', suffix='__raw')
    p, p_df3, log = pipe(df)

    print(df)
    print(p_df0)
    print(p_df1)
    print(p_df2)
    print(p_df3)

