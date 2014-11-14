(ns clj_naive_bayes.core
  (:use [clj_naive_bayes.utils])
  (:require [clojure.core.memoize :as memo]))

(def ^:dynamic classifier nil)

(defn new-classifier
  ([]
   (new-classifier {:name :multinomial-nb}))
  ([algorithm]
    (atom {:all {:tokens {} :n 0 :v 0} :classes {} :algorithm algorithm})))

(defmacro with-classifier
  "Executes body using passed classifier"
  [classifier & body]
    `(binding [classifier ~classifier]
       ~@body))

(defn prior
  "Calculates the prior propability of class c for given classifier"
  [classifier c]
  (/ (get-in @classifier [:classes c :n] 0)
     (get-in @classifier [:all :n])))
(def memo-prior (memoize prior))

(defn Tct
  "Gets the occurences of token t in class c for given classifier"
  [classifier t c]
  (get-in @classifier [:classes c :tokens t] 0))

(defn sum-of-tokens
  "Calculates the sum of all tokens frequencies for class c in a given
  classifier"
  [classifier c]
  (reduce + (map val (get-in @classifier [:classes c :tokens]))))
(def memo-sum-of-tokens (memoize sum-of-tokens))

(defn all-vocabulary
  "Gets all total known vocabulary for a classifier"
  [classifier]
  (get-in @classifier [:all :v] 0))
(def memo-all-vocabulary (memoize all-vocabulary))

(defn condprob
  "Calculates the conditional propability of token t for class c in a
  given classifier"
  [classifier t c]
  (/ (inc (Tct classifier t c))
     (+ (memo-sum-of-tokens classifier c) (memo-all-vocabulary classifier))))

(defn classifier-classes
  "Gets all classes for a given classifier"
  [classifier]
  (keys (get-in @classifier [:classes])))
(def memo-classifier-classes (memoize classifier-classes))

(defn apply-nb
  [classifier document]
  (let [classes (memo-classifier-classes classifier)
        with-algorithm (@classifier :algorithm)
        tokens (flatten (process-features document with-algorithm))]
    (apply hash-map (flatten (map (fn [klass]
          [klass (+ (Math/log (memo-prior classifier klass))
                    (reduce + (map #(Math/log (condprob classifier % klass)) tokens)))])
         classes)))))

(defn classify
  [classifier document]
  ((first (sort-by val > (apply-nb classifier document))) 0))

(defn debug-classify
  [classifier document]
  (sort-by val > (apply-nb classifier document)))