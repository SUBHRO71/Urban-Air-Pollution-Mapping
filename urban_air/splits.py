from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class Split:
    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

def inner_split(indices, horizon, minimum=24):
    size=max(minimum, int(len(indices)*.1))
    cut=len(indices)-size
    train=indices[:cut-horizon]
    validation=indices[cut:]
    if len(train)<48 or len(validation)<minimum:
        raise ValueError("Insufficient history for training, validation and the horizon purge.")
    return train,validation

def make_splits(n, ratio, horizon, evaluation):
    cutoff=int(n*ratio);prefix=np.arange(max(0,cutoff-horizon))
    result=[]
    if evaluation in {"holdout","both"}:
        train,val=inner_split(prefix,horizon)
        if n-cutoff<24:raise ValueError("Holdout needs at least 24 forecast origins.")
        result.append(Split("Holdout",train,val,np.arange(cutoff,n)))
    if evaluation in {"cv","both"}:
        development=np.arange(cutoff)
        try:
            test_size=(len(development)-max(200,len(development)//3))//10
            if test_size<24:
                raise ValueError("Ten evaluation blocks need at least 24 origins each after the initial training block.")
            for i,(outer,test) in enumerate(TimeSeriesSplit(n_splits=10,gap=horizon,test_size=test_size).split(development),1):
                if len(outer)<168 or len(test)<24:
                    raise ValueError("Insufficient history for 10 folds: first training prefix needs 168 origins and each evaluation block 24. Increase the date range or quick-check limit.")
                train,val=inner_split(outer,horizon)
                result.append(Split(f"Fold {i}",train,val,test))
        except ValueError as exc:
            raise ValueError(f"Cannot create ten chronological folds: {exc}") from exc
    return result,cutoff
