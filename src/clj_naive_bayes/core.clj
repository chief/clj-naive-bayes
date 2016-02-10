(ns clj_naive_bayes.core
  (:use [clj_naive_bayes.utils])
  (:require [clojure.core.memoize :as memo]
            [schema.core :as s]))

(defmacro with-classifier
  "Executes body using passed classifier"
  [classifier & body]
  `(binding [classifier ~classifier]
     ~@body))

(defrecord Classifier
    [all
     classes
     algorithm])

(defn new-classifier
  ([]
   (new-classifier {:name :multinomial-nb}))
  ([algorithm]
   (->Classifier (atom {:tokens {} :n 0 :v 0 :st 0}) (atom {})  algorithm)))

(defn Nc
  "Gets the total documents of class c"
  [classifier c]
  (get-in @(:classes classifier) [c :n] 0))

(defn N
  "Gets total documents"
  [classifier]
  (get-in @(:all classifier) [:n]))

(defn prior
  "Calculates the prior propability of class c"
  [classifier c]
  (/ (Nc classifier c) (N classifier)))

(s/defn Tct :- s/Num
  "The number of occurrences of t in training documents from class c"
  [classifier t c]
  (get-in @(:classes classifier) [c :tokens t] 0))

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
  (fn [classifier t c] (get-in classifier [:algorithm :name])))

(defmethod condprob :default
  [classifier t c]
  (/ (inc (Tct classifier t c))
     (+ (NTct classifier c) (B classifier))))

(defmethod condprob :bernoulli
  [classifier t c]
  (/ (inc (Tct classifier t c))
     (+ (Nc classifier c) 2)))

(s/defn Nt :- s/Num
  "Get the occurences of token t in all classes"
  [classifier t]
  (get-in @(:all classifier) [:tokens t] 0))

(s/defn NCt :- s/Num
  "Gets the occurences of token t in  all classes except c"
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

(defn complement-naive-bayes
  "Calculates the Complement Naive Bayes (CNB) of token t for class c"
  [classifier t c]
  (/ (inc (NCt classifier t c))
     (+ (NC classifier c) (B classifier))))

(defn classifier-classes
  "Gets all classes"
  [classifier]
  (keys @(:classes classifier)))

(defn apply-nb
  [classifier document]
  (let [classes (classifier-classes classifier)
        with-algorithm (:algorithm classifier)
        tokens (flatten (process-features document with-algorithm))]
    (apply hash-map
           (flatten (map (fn [klass]
                           [klass (+ (Math/log (prior classifier klass))
                                     (reduce + (map #(Math/log (condprob classifier % klass)) tokens)))])
                         classes)))))

(defn apply-cnb
  [classifier document]
  (let [classes (classifier-classes classifier)
        with-algorithm (:algorithm classifier)
        tokens (flatten (process-features document with-algorithm))]
    (apply hash-map
           (flatten (map (fn [klass]
                           [klass (- (Math/log (prior classifier klass))
                                     (reduce + (map
                                                #(Math/log
                                                  (complement-naive-bayes classifier % klass))
                                                tokens)))])
                         classes)))))

(defn apply-one-versus-all-but-one
  [classifier document]
  (let [classes (classifier-classes classifier)
        with-algorithm (:algorithm classifier)
        tokens (flatten (process-features document with-algorithm))]

    (apply hash-map
           (flatten (map (fn [klass]
                           [klass (+ (Math/log (prior classifier klass))
                                     (reduce + (map
                                                #(- (Math/log
                                                     (condprob classifier % klass))

                                                    (Math/log
                                                     (complement-naive-bayes classifier % klass))

                                                    )
                                                tokens) ))])
                         classes)))))

(defn classify
  [classifier document]
  ((first (sort-by val > (apply-nb classifier document))) 0))

(defn classify-cnb
  [classifier document]
  ((first (sort-by val > (apply-cnb classifier document))) 0))

(defn classify-one-versus-all-but-one
  [classifier document]
  (try
    ((first (sort-by val > (apply-one-versus-all-but-one classifier document))) 0)
    (catch Exception e (str "caught exception: " (.getMessage e) " " document))))

(defn debug-classify
  [classifier document]
  (sort-by val > (apply-nb classifier document)))
