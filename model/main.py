import pandas as pd
import psycopg2
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
    cursor.execute("SELECT o.name as name, m.id as id, mce.name as mname" +
      " from measurement m, outcome o, study_design sd, sd_on_use sdou, clinical_entity mce" +
      " where m.id=o.measure_id" +
      " AND m.id = mce.measurement_id" +
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
  parser.add_argument('--sentence', default=None)
  args = parser.parse_args()

  es = Ensemble_Model()

  if args.filename is not None and args.sentence is not None:
    es.load(args.filename)
    prediction = es.predict(preprocess_sentence(args.sentence), "")
    print("Prediction: {}".format(prediction))

  else:
    df = pd.DataFrame(get_data(args.samp_size))
    sentences, topics, m_names = preprocess(df.dropna())
    es.fit(sentences, topics, m_names)
    es.save('./saved_models/{}'.format(es.id))
