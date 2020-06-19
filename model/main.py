import argparse
from model import *
import pandas as pd
import psycopg2
from utils import *

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
    cursor.execute("SELECT concat(o.name, '. ', o.description) as name, m.id as id" +
      " from measurement m, outcome o, clinical_entity mce" +
      " where m.id=o.measure_id" +
      " AND m.id = mce.measurement_id" +
      " AND o.disabled = false" +
      " AND mce.is_disabled = false" +
      " AND mce.is_provisional = false" +
      " ORDER BY random()" +
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
    prediction = es.predict(preprocess_sentence(args.sentence))
    print("Prediction: {}".format(prediction))

  else:
    df = pd.DataFrame(get_data(args.samp_size))
    text, topics = preprocess(df.dropna())
    es.fit(text, topics)
    es.save('./saved_models/{}'.format(es.id))
