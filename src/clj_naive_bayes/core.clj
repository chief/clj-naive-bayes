(ns clj_naive_bayes.core
  (:require [schema.core :as s]
            [clj_naive_bayes.utils :as utils]))

(defmacro with-classifier
  "Executes body using passed classifier"
  [classifier & body]
  `(binding [classifier ~classifier]
     ~@body))

(s/defrecord Classifier [algorithm :- {}
                         data :- s/atom
                         score])

;; Data representation:
;; --------------------
;; all:
;;   n: document count
;;   v: unique token count
;;   st: raw token count (including duplicates)
;; tokens:
;;   all: raw token count (including duplicates)
;;   class_name: times token was observed in this class (including duplicates)
;; classes:
;;   n: documents in that class
;;   st: raw token count in that class (including duplicates)

(defn new-classifier
  ([]
   (new-classifier {:name :multinomial-nb} :default))
  ([algorithm score]
   (->Classifier algorithm
                 (atom {:all {:n 0 :v 0}
                        :classes {}
                        :tokens {}})
                 score)))

(defn Nc
  "Gets total documents of class c"
  [classifier c]
  (get-in @(:data classifier) [:classes c :n] 0))

(defn N
  "Gets total documents for all classes"
  [classifier]
  (get-in @(:data classifier) [:all :n]))

(defn prior
  "Calculates the prior propability of class c"
  [classifier c]
  (/ (Nc classifier c) (N classifier)))

(s/defn Tct :- s/Num
  "The number of occurrences of t in training documents from class c"
  [classifier t c]
  (get-in @(:data classifier) [:tokens t c] 0))

(s/defn NTct :- s/Num
  "Gets total token occurrences for a class c"
  [classifier c]
  (get-in @(:data classifier) [:classes c :st] 0))

(s/defn B :- s/Num
  "or |V| is the number of terms in the vocabulary"
  [classifier]
  (get-in @(:data classifier) [:all :v] 0))

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
  (get-in @(:data classifier) [:tokens t :all] 0))

(s/defn NCt :- s/Num
  "Gets the occurences of token t in all classes except c"
  [classifier t c]
  (- (Nt classifier t) (Tct classifier t c)))

(s/defn Nst :- s/Num
  "Gets total token occurences for a classifier"
  [classifier]
  (get-in @(:data classifier) [:all :st] 0))

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
  (keys (:classes @(:data classifier))))

(defmulti score
  (fn [classifier tokens klass] (:score classifier)))

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
  (+ (- (Math/log (prior classifier klass))
        (Math/log (- 1 (prior classifier klass))))
     (reduce + (map
                #(- (Math/log (condprob classifier % klass))
                    (Math/log (cnb-condprob classifier % klass))) tokens))))

(defn export-multinomial-nested [classifier]
  {:tokens (apply merge-with merge
                  (for [[t cats] (:tokens @(:data classifier))
                        [cid _] cats
                        :when (not (= :all cid))]
                    {t {cid (Math/log (condprob classifier t cid))}}))
   :classes (reduce (fn [coll cid]
                      (assoc coll cid
                             {:prior (Math/log (prior classifier cid))
                              :flat-prior (Math/log (condprob classifier cid))}))
                    {} (classifier-classes classifier))})

;; TODO: Use this to export to CSV file (and maybe create a helper
;; export-to-file function.
(defn export-multinomial-flat  [classifier]
  {:tokens (for [[t cats] (:tokens @(:data classifier))
                 [cid _] cats
                 :when (not (= :all cid))]
             [t cid (Math/log (condprob classifier t cid))])
   :classes (map (fn [c] [c
                          (Math/log (prior classifier c))
                          (Math/log (condprob classifier c))])
                 (classifier-classes classifier))})

;; TODO: this only works for multinomial at the moment.
(defmulti export
  (fn [classifier output] (get-in classifier [:algorithm :name])))

(defmethod export :default
  [classifier output]
  (condp = output
    :nested (export-multinomial-nested classifier)
    :flat (export-multinomial-flat classifier)
    (export-multinomial-nested classifier)))

(defn score-document
  [classifier document]
  (let [classes (classifier-classes classifier)
        tokens (utils/process-features classifier document)]
    (map (fn [c] [c (score classifier tokens c)]) classes)))

(defn classify [classifier document]
  (first (apply max-key second (score-document classifier document))))

(defn top-classes [classifier document n threshold]
  (->> (score-document classifier document)
       (remove #(< (second %) threshold))
       (sort-by second >)
       (take n)))
