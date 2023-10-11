from time import sleep

from pyspark import SparkContext

sc = SparkContext("local", "count app")
words = sc.parallelize (
   ["scala",
   "java",
   "hadoop",
   "spark",
   "akka",
   "spark vs hadoop",
   "pyspark",
   "pyspark and spark"], 3
)

print(type(words))
counts = words.count()
print ("Number of elements in RDD -> %i" % (counts))

coll = words.collect()
print ("Elements in RDD -> %s" % (coll))

def f(x): print(x)
fore = words.foreach(f)

words_filter = words.filter(lambda x: 'spark' in x)
filtered = words_filter.collect()
print ("Fitered RDD -> %s" % (filtered))

sleep(100)