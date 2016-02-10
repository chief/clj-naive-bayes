(ns clj_naive_bayes.utils
  (:use [clojure.java.io :only (reader)])
  (:require [cheshire.core :refer :all]
            [clojure.data.csv :as csv]
            [clojure.edn :as edn]
            [schema.core :as s]))

(defn load-data
  [filename]
  (with-open [in-file (reader filename)]
    (doall
     (csv/read-csv in-file))))

(defn tokenize
  "Returns a tokenized string"
  [s]
  (clojure.string/split s #"\s+"))

(s/defn build-partitions :- []
  [array size explode]
  (if explode
    (map #(partition % 1 array) (range 1 (inc size)))
    (partition size 1 array)))

(defn ngram-keys
  "Returns ngram keys from an array of tokens."
  [array & {:keys [size type explode-ngrams]
            :or {size 2
                 type :multinomial
                 explode-ngrams false}}]
  (let [ngrams (map #(clojure.string/join "_" %) (build-partitions array size explode-ngrams))]
    (if (= type :binary)
      (distinct ngrams)
      ngrams)))

(defmulti process-features
  "Process features based on Classifier algorithm"
  (fn [classifier features] (get-in classifier [:algorithm :name])))

(s/defmethod process-features :multinomial-nb :- []
  [classifier features]
  (map tokenize features))

(defmethod process-features :binary-nb
  [classifier features]
  (map #(distinct (tokenize %)) features))

(defmethod process-features :bernoulli
  [classifier features]
  (->> features
       (map tokenize)
       flatten
       distinct))

(defmethod process-features :ngram-nb
  [classifier features]
  (let [ngram-size (get-in classifier [:algorithm :ngram-size] 2)
        ngram-type (get-in classifier [:algorithm :ngram-type] :multinomial)
        explode-ngrams (get-in classifier [:algorithm :explode-ngrams] false)]
    (->> features
         (map #(ngram-keys (tokenize %) :size ngram-size :type ngram-type
                           :explode-ngrams explode-ngrams)))))

(defn persist-classifier
  [classifier filename]
  (with-open [w (clojure.java.io/writer filename)]
    (binding [*out* w]
      (pr (deref classifier)))))

(defn- read-classifier-data
  [file]
  (with-open [r (java.io.PushbackReader. (clojure.java.io/reader file))]
    (binding [*read-eval* false]
      (read r))))

(defn load-classifier
  [classifier file]
  (let [x (read-classifier-data file)]
    (swap! classifier assoc :all (:all x))
    (swap! classifier assoc :classes (:classes x))
    (println "loading complete.")))

(defn repl-init
  []
  (use 'clj_naive_bayes.core :reload-all)
  (use 'clj_naive_bayes.eval :reload-all)
  (use 'clj_naive_bayes.train :reload-all)
  (use 'clojure.tools.namespace.repl)
  (require 'spyscope.core)
  (require 'clojure.pprint))
