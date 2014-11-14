(ns clj_naive_bayes.utils
  (:use [clojure.java.io :only (reader)])
  (:require [cheshire.core :refer :all]
            [clojure.data.csv :as csv]
            [clojure.edn :as edn]))

(defn load-data
  [filename]
  (with-open [in-file (reader filename)]
    (doall
      (csv/read-csv in-file))))

(defn tokenize
  "Returns a tokenized string"
  [s]
  (clojure.string/split s #"\s+"))

(defn tokenize-and-set
  "Tokenize a string and returns a set"
  [s]
  (set (tokenize s)))

(defn greek-books-or-empty-category?
  [cid]
  (or (= cid 34) (zero? cid)))

(defn ngram-keys
  "Given an array returns an"
  [array & {:keys [ngram-size]
            :or {ngram-size 2}}]
  (map #(clojure.string/join "_" %) (partition ngram-size 1 array)))

(defn process-features
  [features for-algorithm]
  (cond
    (= for-algorithm :multinomial-nb)
      (map tokenize features)
    (= for-algorithm :binary-nb)
      (map #(distinct (tokenize %)) features)
    (= for-algorithm :ngram-nb)
      (map #(ngram-keys (tokenize %)) features)))

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
  (require 'clojure.pprint))