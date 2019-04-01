from typing import Any, Dict, Generator, List

from toolz.curried import reduce, partial, pipe, first, curry

from fklearn.metrics.pd_extractors import extract
from fklearn.types import LogListType, LogType, ExtractorFnType, ValidatorReturnType, EvalReturnType


@curry
def get_avg_metric_from_extractor(logs: LogType, extractor: ExtractorFnType, metric_name: str) -> float:
    metric_folds = extract(logs["validator_log"], extractor)
    return metric_folds[metric_name].mean()


def get_best_performing_log(log_list: LogListType, extractor: ExtractorFnType, metric_name: str) -> Dict:
    logs_eval = [get_avg_metric_from_extractor(log, extractor, metric_name) for log in log_list]
    return pipe(logs_eval, partial(zip, log_list), partial(sorted, reverse=True, key=lambda x: x[1]))[0][0]


def get_used_features(log: Dict) -> List[str]:
    return first((gen_dict_extract('features', log)))


def order_feature_importance_avg_from_logs(log: Dict) -> List[str]:
    d = first(gen_dict_extract('feature_importance', log))
    return sorted(d, key=d.get, reverse=True)


def gen_key_avgs_from_logs(key: str, logs: List[Dict]) -> Dict[str, float]:
    return gen_key_avgs_from_dicts([gen_key_avgs_from_iteration(key, log) for log in logs])


def gen_key_avgs_from_iteration(key: str, log: Dict) -> Any:
    return first(gen_dict_extract(key, log))


def gen_key_avgs_from_dicts(obj: List) -> Dict[str, float]:
    sum_values_by_key = reduce(lambda x, y: dict((k, v + y.get(k, 0)) for k, v in x.items()), obj)
    return {k: float(v) / len(obj) for k, v in sum_values_by_key.items()}


def gen_dict_extract(key: str, obj: Dict) -> Generator[Any, None, None]:
    if hasattr(obj, 'items'):
        for k, v in obj.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


@curry
def gen_validator_log(eval_log: EvalReturnType, fold_num: int, test_size: int) -> ValidatorReturnType:
    return {'validator_log': [{'fold_num': fold_num, 'split_log': {'test_size': test_size},
                               'eval_results': [eval_log]}]}
