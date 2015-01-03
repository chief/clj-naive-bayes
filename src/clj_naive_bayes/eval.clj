(ns clj_naive_bayes.eval
  (:use [clj_naive_bayes.core]
        [clojure.java.io :only (reader)])
  (:require [cheshire.core :refer :all]))

(defn parallel-classifications
  [classifier with-file & {:keys [fields limit]
                 :or {fields ["item" "shopcategory"]
                      limit 100}}]
  (let [logs (parsed-seq (reader with-file))]
    (dosync
      (pmap #(vector (% "actual_prediction")
                     (vals (select-keys % fields))
                     (classify classifier (vals (select-keys % fields))))
       (take limit logs)))))

(defn parallel-classifications-cnb
  [classifier with-file & {:keys [fields limit]
                 :or {fields ["item" "shopcategory"]
                      limit 100}}]
  (let [logs (parsed-seq (reader with-file))]
    (dosync
      (pmap #(vector (% "actual_prediction")
                     (vals (select-keys % fields))
                     (classify-cnb classifier (vals (select-keys % fields))))
       (take limit logs)))))

(defn parallel-classifications-ovabo
  [classifier with-file & {:keys [fields limit]
                 :or {fields ["item" "shopcategory"]
                      limit 100}}]
  (let [logs (parsed-seq (reader with-file))]
    (dosync
      (pmap #(vector (% "actual_prediction")
                     (vals (select-keys % fields))
                     (classify-one-versus-all-but-one classifier (vals (select-keys % fields))))
       (take limit logs)))))

(defn parallel-classifications-v2
  [classifier with-file & {:keys [fields limit]
                 :or {fields ["item" "shopcategory"]
                      limit 100}}]
  (let [logs (parsed-seq (reader with-file))]
    (dosync
      (pmap #(vector (% "actual_prediction")
                     (vals (select-keys % fields))
                     (map first (take 3 (debug-classify classifier (vals (select-keys % fields))))))
       (take limit logs)))))

(defn evaluate-current-algorithm
  [against-classifier with-file]
  (let [x (parallel-classifications)]
    (count (filter #(= (keyword (str (first %))) (last %)) x))))

(defn evaluate-performance
  [filename actual pred limit]
  (let [logs (take limit (parsed-seq (reader filename)))
        n (count logs)
        correct (count (filter #(= (% actual) (% pred)) logs))
        ; no-prediction (count (filter #(zero? (% pred)) logs))]
        ]
    (println (count logs))
    {:accuracy (float (/ correct n))
     ; :precision (float (/ correct (- n no-prediction)))
     }))

(defn take-failed
  [filename limit actual pred]
  (let [logs (take limit (filter #(not= (% actual) (% pred))
                                  (parsed-seq (reader filename))))]
    logs))