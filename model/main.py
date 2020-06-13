import pandas as pd
import pickle
import psycopg2
import matplotlib.pyplot as plt
from model import *
from utils import *

import argparse

# load data from postgres
def get_data(limit = 1000):
  try:
    connection = psycopg2.connect(
      user = "kristinlindquist",
      password = "",
      host = "127.0.0.1",
      port = "5432",
      database = "rationally"
    )

    cursor = connection.cursor()
    cursor.execute("SELECT o.name as name, m.id as id" +
      " from measurement m, outcome o, study_design sd, sd_on_use sdou" +
      " where m.id=o.measure_id" +
      " AND COALESCE(o.po_parent_id, o.so_parent_id)=sd.id" +
      " AND sdou.study_design_id=sd.id AND sdou.use_id=148" +
      " limit " + str(limit))
    records = cursor.fetchall()

  except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
  finally:
    if(connection):
      cursor.close()
      connection.close()
      print("PostgreSQL connection is closed")
  return records


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--samp_size', default=10000)
  parser.add_argument('--filename', default=None)
  args = parser.parse_args()

  df = pd.DataFrame(get_data(args.samp_size))
  sentences, tokens, topics = preprocess(df.dropna())

  if args.filename is not None:
    es = pickle.load(open(args.filename, 'rb'))
    es.score(sentences, tokens, topics)

  else:
    es = Ensemble_Model()
    es.fit(sentences, tokens, topics)

    with open("./saved_models/{}.file".format(es.id), "wb") as f:
      pickle.dump(es, f, pickle.HIGHEST_PROTOCOL)
