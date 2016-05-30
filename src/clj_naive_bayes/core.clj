(ns clj_naive_bayes.core
  (:require [schema.core :as s]
            [clj_naive_bayes.utils :as utils]))

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

(defn Nc
  "Gets total documents of class c"
  [classifier c]
  (get-in @(:classes classifier) [c :n] 0))

(defn N
  "Gets total documents for all classes"
  [classifier]
  (get-in @(:all classifier) [:n]))

(defn prior
  "Calculates the prior propability of class c"
  [classifier c]
  (/ (Nc classifier c) (N classifier)))

(s/defn Tct :- s/Num
  "The number of occurrences of t in training documents from class c"
  [classifier t c]
  (get-in @(:tokens classifier) [t c] 0))

(s/defn NTct :- s/Num
  "Gets total token occurrences for a class c"
  [classifier c]
  (get-in @(:classes classifier) [c :st] 0))

(s/defn B :- s/Num
  "or |V| is the number of terms in the vocabulary"
  [classifier]
  (get-in @(:all classifier) [:v] 0))

(defmulti condprob
  "Calculates the conditional propability of token t for class c"
  (fn [classifier & args] (get-in classifier [:algorithm :name])))

(defmethod condprob :default
  ([classifier c]
   (/ 1
      (+ (NTct classifier c) (B classifier))))
  ([classifier t c]
   (/ (inc (Tct classifier t c))
      (+ (NTct classifier c) (B classifier)))))

(defmethod condprob :bernoulli
  [classifier t c]
  (/ (inc (Tct classifier t c))
     (+ (Nc classifier c) 2)))

(s/defn Nt :- s/Num
  "Gets the occurences of token t in all classes"
  [classifier t]
  (get-in @(:tokens classifier) [t :all] 0))

(s/defn NCt :- s/Num
  "Gets the occurences of token t in all classes except c"
  [classifier t c]
  (- (Nt classifier t) (Tct classifier t c)))

(s/defn Nst :- s/Num
  "Gets total token occurences for a classifier"
  [classifier]
  (get-in @(:all classifier) [:st] 0))

(s/defn NC :- s/Num
  "Gets total number of token occurrences in classes other than c"
  [classifier c]
  (- (Nst classifier) (NTct classifier c)))

(defn cnb-condprob
  "Calculates the Complement Naive Bayes (CNB) condprob of token t for class c"
  [classifier t c]
  (/ (inc (NCt classifier t c))
     (+ (NC classifier c) (B classifier))))

(defn classifier-classes
  "Gets all classes"
  [classifier]
  (keys @(:classes classifier)))

(defmulti score
  (fn [classifier tokens klass] @(get classifier :score)))

(defmethod score :default
  [classifier tokens klass]
  (+  (Math/log (prior classifier klass))
      (reduce + (map #(Math/log (condprob classifier % klass)) tokens))))

(defmethod score :complement-naive-bayes
  [classifier tokens klass]
  (- (Math/log (prior classifier klass))
     (reduce + (map  #(Math/log (cnb-condprob classifier % klass)) tokens))))

(defmethod score :one-versus-all-but-one
  [classifier tokens klass]
  (+ (Math/log (prior classifier klass))
     (reduce + (map
                #(- (Math/log (condprob classifier % klass))
                    (Math/log (cnb-condprob classifier % klass))) tokens))))

(defmulti export
  (fn [classifier] (get-in classifier [:algorithm :name])))

;; TODO: this only works for multinomial at the moment.
(defmethod export :default
  [classifier]
  {:terms (for [[t cats] @(:tokens classifier)
                [cid _] cats
                :when (not (= :all cid))]
            [t cid (Math/log (condprob classifier t cid))])
   :cats (map (fn [c] [c
                       (Math/log (prior classifier c))
                       (Math/log (condprob classifier c))])
              (keys @(:classes classifier)))})

(defn apply-nb
  [classifier document]
  (let [classes (classifier-classes classifier)
        tokens (utils/process-features classifier document)]
    (reduce into {} (map #(hash-map % (score classifier tokens %)) classes))))

(defn classify
  [classifier document]
  ((first (sort-by val > (apply-nb classifier document))) 0))

(defn best-n-classes
  [classifier document n]
  (take n (sort-by val > (apply-nb classifier document))))

(defn debug-classify
  ([classifier document]
   (sort-by val > (apply-nb classifier document)))
  ([classifier document n]
   (take n (sort-by val > (apply-nb classifier document)))))
