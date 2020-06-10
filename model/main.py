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
    cursor.execute("SELECT o.name as name, m.id as id from measurement m, outcome o" +
      " where m.id=o.measure_id limit " + str(limit))
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
    tm = pickle.load(open(args.filename, 'rb'))
    tm.score(sentences, tokens, topics)

  else:
    k = pd.unique(topics).size
    tm = Topic_Model(k = k)
    tm.fit(sentences, tokens, topics)

    with open("./saved_models/{}.file".format(tm.id), "wb") as f:
      pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)
  # df = pd.DataFrame(get_data())[0];
  # sentences, token_lists = preprocess(df, samp_size=int(args.samp_size))


  # print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
  # print('Silhouette Score:', get_silhouette(tm))

  # visualize(tm)
  # for i in range(tm.k):
  #   get_wordcloud(tm, token_lists, i)
  # df.sample(n = 10).apply(
  #   lambda row: print("{}: {}, previously {}".format(row[0], tm.predict(row[0], token_lists), row[1])),
  #   axis=1
  # )