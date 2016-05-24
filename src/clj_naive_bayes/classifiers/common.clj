(ns clj-naive-bayes.classifiers.common
  (:require [schema.core :as s]))

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

(s/defn Nc :- s/Num
  "Gets total documents of class c"
  [classifier c]
  (get-in @(:classes classifier) [c :n] 0))

(s/defn N :- s/Num
  "Gets total documents for all classes"
  [classifier]
  (get-in @(:all classifier) [:n]))

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

(s/defn prior :- s/Num
  "Calculates the prior propability of class c"
  [classifier c]
  (/ (Nc classifier c) (N classifier)))

;; TODO: Is this still needed? Should each classifier be responsible for this?
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

(defn cnb-condprob
  "Calculates the Complement Naive Bayes (CNB) condprob of token t for class c"
  [classifier t c]
  (/ (inc (NCt classifier t c))
     (+ (NC classifier c) (B classifier))))

(defmulti score
  (fn [classifier tokens klass] @(get classifier :score)))

(defmethod score :default
  [classifier tokens klass]
  (+ (Math/log (prior classifier klass))
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
