In og code: missing: null for year. Is not possible anymore (throws error), so instead: missing 0? [[QUESTION CLAUDIA]] if that is appropriate.



`b = {'aggs':{'miss':{'filter': {'ids':{'values': ll}},'aggs':{'miss':{'missing':{'field':'year'}}}}}}`
--> ` 'aggregations': {'miss': {'doc_count': 3571, 'miss': {'doc_count': 31}}}}`

31 van de training documents hebben geen year 




WE STICK WITH 0 for now then i guess