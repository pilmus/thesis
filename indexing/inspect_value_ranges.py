from elasticsearch import Elasticsearch
from elasticsearch_dsl import AttrDict

es = Elasticsearch()

q = {
    "size": 0,
    "aggs": {
        "year_agg": {
            "terms": {"field": "year", "size": 500}
        },
        "num_in_agg" : {
            "terms": {"field" : "inCitations", "size": 500}
        },
        "num_out_agg" : {
            "terms": {"field" : "outCitations", "size": 500}
        }
    }}

res = es.search(index='semanticscholar2019og', body=q)
res = AttrDict(res)

year_range = sorted([r.key for r in res.aggregations.year_agg.buckets])
in_range = sorted([r.key for r in res.aggregations.num_in_agg.buckets])
out_range = sorted([r.key for r in res.aggregations.num_out_agg.buckets])

print(year_range[0], year_range[-1])
print(in_range[0], in_range[-1])
print(out_range[0], out_range[-1])
