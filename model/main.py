import pandas as pd
import pickle
import psycopg2
import matplotlib.pyplot as plt
from model import *
from utils import *

import argparse

# load data from postgres
def get_data():
  try:
    connection = psycopg2.connect(
      user = "kristinlindquist",
      password = "",
      host = "127.0.0.1",
      port = "5432",
      database = "rationally"
    )

    cursor = connection.cursor()
    cursor.execute("SELECT concat(sd.name, ' ', title, ' ', summary) from study_design sd, meta, sd_on_use sdou" +
      " where sd.id=meta.study_design_id AND sdou.study_design_id=sd.id AND sdou.use_id=8280")
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
    parser.add_argument('--ntopic', default=5)
    parser.add_argument('--samp_size', default=10000)
    args = parser.parse_args()

    data = pd.DataFrame(get_data())[0];
    sentences, token_lists = preprocess(data, samp_size=int(args.samp_size))

    tm = Topic_Model(k = int(args.ntopic))
    tm.fit(sentences, token_lists)

    with open("./saved_models/{}.file".format(tm.id), "wb") as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))

    visualize(tm)
    for i in range(tm.k):
      get_wordcloud(tm, token_lists, i)
