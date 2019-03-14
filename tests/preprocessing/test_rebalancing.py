from fklearn.preprocessing.rebalancing import rebalance_by_categorical, rebalance_by_continuous
import pandas as pd

data1 = pd.DataFrame({"col1": ["a", "b", "c", "a", "b", "c", "a", "a", "b", "b", "b"]})
data2 = pd.DataFrame({"col2": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5]})


def test_rebalance_by_categorical():
    result1 = rebalance_by_categorical(data1, "col1").sort_values(by="col1")

    expected1 = pd.DataFrame({
        'col1': ["a", "a", "b", "b", "c", "c"],
    })

    assert result1.reset_index(drop=True).equals(expected1), "not working with non numeric column"

    result2 = rebalance_by_categorical(data2, "col2", max_lines_by_categ=2).sort_values(by="col2")

    expected2 = pd.DataFrame({
        "col2": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    })

    assert result2.reset_index(drop=True).equals(expected2), "not working with numeric columns"


def test_rebalance_by_continuous():

    result1 = (rebalance_by_continuous(data2, "col2", buckets=3, by_quantile=False)
               .sort_values("col2")
               .reset_index(drop=True))

    expected1 = pd.DataFrame({
        "col2": [2, 1, 1, 3, 3, 3, 4, 5, 5],
    }).sort_values("col2").reset_index(drop=True)

    assert result1.reset_index(drop=True).equals(expected1), "not working with pd.cut"

    result1 = (rebalance_by_continuous(data2, "col2", buckets=3, by_quantile=True)
               .sort_values("col2")
               .reset_index(drop=True))

    expected1 = pd.DataFrame({
        "col2": [2, 1, 1, 2, 1, 1, 1, 2, 4, 5, 5, 3, 5, 4, 3, 5],
    }).sort_values("col2").reset_index(drop=True)

    assert result1.equals(expected1), "not working with pd.qcut"
