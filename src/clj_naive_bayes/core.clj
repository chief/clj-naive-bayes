(ns clj_naive_bayes.core
  (:use [clj_naive_bayes.utils])
  (:require [clojure.core.memoize :as memo]))

(def ^:dynamic classifier nil)

(defn new-classifier
  ([]
   (new-classifier {:name :multinomial-nb}))
  ([algorithm]
    (atom {:all {:tokens {} :n 0 :v 0 :st 0} :classes {} :algorithm algorithm})))

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

(defn Tct
  "Gets the occurences of token t in class c for given classifier"
  [classifier t c]
  (get-in @classifier [:classes c :tokens t] 0))

(defn Nstc
  "Gets total token occurrences for a class c"
  [classifier c]
  (get-in @classifier [:classes c :st] 0))

(defn all-vocabulary
  "Gets all total known vocabulary for a classifier"
  [classifier]
  (get-in @classifier [:all :v] 0))

(defn condprob
  "Calculates the conditional propability of token t for class c in a
  given classifier"
  [classifier t c]
  (/ (inc (Tct classifier t c))
     (+ (Nstc classifier c) (all-vocabulary classifier))))

(defn Nt
  "Get the occurences of token t in all classes for a given classifier"
  [classifier t]
  (get-in @classifier [:all :tokens t] 0))

(defn NCt
  "Gets the occurences of token t in  all classes except c for a given classifier"
  [classifier t c]
  (- (Nt classifier t) (Tct classifier t c)))

(defn Nst
  "Gets total token occurences for a classifier"
  [classifier]
  (get-in @classifier [:all :st] 0))

(defn Nc
  "Gets total number of word occurrences in classes other than c for a given
  classifier"
  [classifier c]
  (- (Nst classifier) (Nstc classifier c)))

(defn complement-naive-bayes
  "Calculates the Complement Naive Bayes (CNB) of token t for class c in a
  given classifier"
  [classifier t c]
   (/ (inc (NCt classifier t c))
     (+ (Nc classifier c) (all-vocabulary classifier))))

(defn classifier-classes
  "Gets all classes for a given classifier"
  [classifier]
  (keys (get-in @classifier [:classes])))

(defn apply-nb
  [classifier document]
  (let [classes (classifier-classes classifier)
        with-algorithm (@classifier :algorithm)
        tokens (flatten (process-features document with-algorithm))]
    (apply hash-map (flatten (map (fn [klass]
          [klass (+ (Math/log (prior classifier klass))
                    (reduce + (map #(Math/log (condprob classifier % klass)) tokens)))])
         classes)))))

(defn apply-cnb
  [classifier document]
  (let [classes (classifier-classes classifier)
        with-algorithm (@classifier :algorithm)
        tokens (flatten (process-features document with-algorithm))]
    (apply hash-map (flatten (map (fn [klass]
          [klass (- (Math/log (prior classifier klass))
                    (reduce + (map
                                #(Math/log
                                   (complement-naive-bayes classifier % klass))
                                tokens)))])
         classes)))))

(defn apply-one-versus-all-but-one
  [classifier document]
  (let [classes (classifier-classes classifier)
        with-algorithm (@classifier :algorithm)
        tokens (flatten (process-features document with-algorithm))]

    (apply hash-map (flatten (map (fn [klass]
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
