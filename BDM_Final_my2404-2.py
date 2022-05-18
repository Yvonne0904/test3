import sys
import csv
import json
import numpy as np
import pandas as pd

from pyproj import Transformer
import shapely
from shapely.geometry import Point

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions  import date_format
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)



df_pattern = spark.read.option("header","true").option("escape", "\"").csv("/tmp/bdm/weekly-patterns-nyc-2019-2020/*")

df_supermarket = pd.read_csv('nyc_supermarkets.csv')
df_supermarket = spark.createDataFrame(df_supermarket)

df_cbg = pd.read_csv("nyc_cbg_centroids.csv")
df_cbg = spark.createDataFrame(df_cbg)

def point_transform(col1,col2):
  t = Transformer.from_crs(4326, 2263)
  tuple1 = t.transform(col1,col2)
  
  return tuple1

transformUdf = F.udf(point_transform,T.ArrayType(T.DoubleType()))
df_cbg_transform = df_cbg.withColumn('point',transformUdf('latitude','longitude')).select('cbg_fips','point').cache()

# Filter with nyc_supermarkets.csv
placeList = df_supermarket.select("safegraph_placekey").rdd.flatMap(lambda x: x).collect()
df_pattern = df_pattern.filter(F.col("placekey").isin(placeList)).select("placekey","poi_cbg","visitor_home_cbgs", date_format("date_range_start", "yyyy-MM").alias("Formatted_start_date"),date_format("date_range_end", "yyyy-MM").alias("Formatted_end_date")).cache()

dateList = ['2019-03','2019-10','2020-03','2020-10']
df_pattern2 = df_pattern.filter(F.col("Formatted_start_date").isin(dateList) | F.col("Formatted_end_date").isin(dateList)).cache()

def extract_date(col1,col2):
  res = ''
  if col1 == col2:
    res = col1 
  
  elif col1 in ['2019-03','2019-10','2020-03','2020-10']:
    res = col1

  elif col2 in ['2019-03','2019-10','2020-03','2020-10']:
    res = col2 
  
  return res

dateUdf = F.udf(extract_date,T.StringType())
df_pattern_date = df_pattern2.withColumn('date',dateUdf('Formatted_start_date','Formatted_end_date')).select('poi_cbg','visitor_home_cbgs','date').cache()

def f_key(col1):
  visitor_home_cbg = json.loads(col1)
  res = []
  for i in visitor_home_cbg:
    res.append(i)
  return res

def f_value(col1):
  visitor_home_cbg = json.loads(col1)
  value_res = []
  for i in visitor_home_cbg.values():
    value_res.append(i)
  return value_res

keyUdf = F.udf(f_key,T.ArrayType(T.StringType()))
valueUdf = F.udf(f_value,T.ArrayType(T.IntegerType()))
df_pattern3 = df_pattern_date.withColumn('res_key',keyUdf("visitor_home_cbgs")).withColumn('res_value',valueUdf("visitor_home_cbgs")).withColumn("new", F.arrays_zip("res_key", "res_value"))\
       .withColumn("new", F.explode("new"))\
       .select('date',"poi_cbg", F.col("new.res_key").alias("home_key"), F.col("new.res_value").alias("people_number")).filter(F.col('home_key').substr(1, 2) == '36').cache()

join_df1 = df_pattern3.join(df_cbg_transform,F.col('poi_cbg')==F.col('cbg_fips'),'inner')\
.select(['date','poi_cbg','home_key','people_number','point'])\
.withColumnRenamed('point','poi_point')\
.join(df_cbg_transform,F.col('home_key')==F.col('cbg_fips'),'inner')\
.withColumnRenamed('point','home_point').select(['date','poi_cbg','people_number','poi_point','home_point']).cache()

def weigted_dis(col1,col2,col3):
  
  weighted_distance = col3*(Point(tuple(col1)).distance(Point(tuple(col2)))/5280)
  
  return weighted_distance


weight_disUdf = F.udf(weigted_dis,T.DoubleType())
distance_res = join_df1.withColumn('weighted_distance',weight_disUdf('poi_point','home_point','people_number')).select(['date','poi_cbg','people_number','weighted_distance']).cache()

grouped_df = distance_res.groupby('date','poi_cbg').agg(
    F.sum('people_number'),
    (F.sum('weighted_distance')/F.sum('people_number')).alias('weighted_distance')
).groupBy("poi_cbg").pivot('date').sum('weighted_distance').withColumnRenamed('poi_cbg','cbg_fips').sort('cbg_fips').write.option("header",True).csv(sys.argv[1])
