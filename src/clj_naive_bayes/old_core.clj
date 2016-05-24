(ns clj-naive-bayes.old-core
  (:require [schema.core :as s]
            [clj-naive-bayes.utils :as utils]))

(defmacro with-classifier
  "Executes body using passed classifier"
  [classifier & body]
  `(binding [classifier ~classifier]
     ~@body))

(s/defrecord Classifier
    [all :- s/atom
     classes :- s/atom
     algorithm :- {}
     tokens :- s/atom
     score :- s/atom])

(defn new-classifier
  ([]
   (new-classifier {:name :multinomial-nb}))
  ([algorithm]
   (->Classifier (atom {:n 0 :v 0}) (atom {})  algorithm
                 (atom {}) (atom :naive-bayes))))

(defn classifier-classes
  "Gets all classes"
  [classifier]
  (keys @(:classes classifier)))

;; (defn apply-nb
;;   [classifier document]
;;   (let [classes (classifier-classes classifier)
;;         tokens (first (utils/process-features classifier document))]
;;     (reduce into {} (map #(hash-map % (score classifier tokens %)) classes))))

;; (defn classify
;;   [classifier document]
;;   ((first (sort-by val > (apply-nb classifier document))) 0))

;; (defn debug-classify
;;   [classifier document]
;;   (sort-by val > (apply-nb classifier document)))
