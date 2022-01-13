import pyterrier as pt

if not pt.started():
  pt.init()

index = pt.IndexFactory.of("./sample_corp_index/data.properties")
print(index.getCollectionStatistics())
#
# print(pt.BatchRetrieve(index).search("marsh"))
