from pycorenlp import StanfordCoreNLP
import json
nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("Obama is the the president of US. Florida is a nice place. It is good. He lives in Florida. Trump is the current president. He owns Trump tower.",
                   properties={
                       'annotators': 'coref',
                       'outputFormat': 'json',
                       'timeout': 75000,
                   })
# print(res)
result = json.loads(res)
print(result['corefs'])