from copy import deepcopy
import json
from decimal import Decimal
from pydantic import BaseModel


class Ranker(BaseModel):
    sub_rank:Decimal

class json_object(BaseModel):
    name:str
    age:int
    rank:Ranker





if __name__=="__main__":
    x0=Ranker(sub_rank=123)
    x1=json_object(name="Alex",age=34,rank=x0)
    x2=deepcopy(x1)
    print(x1)
    x2.name="Samson"
    print(x1)
    print(x2.rank.sub_rank)