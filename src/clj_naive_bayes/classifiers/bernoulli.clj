(ns clj-naive-bayes.classifiers.bernoulli
  (:require [clj-naive-bayes.protocols :refer :all]
            [clj-naive-bayes.classifiers.multinomial :as multinomial]
            [clj-naive-bayes.classifiers.common :as common]
            [clj-naive-bayes.utils :as utils]
            [schema.core :as s]))

(defn- preprocess-document [document]
  (->> (utils/tokenize document)
       distinct))

;; Training phase is exactly the same as the in the Multinomial case so we
;; use the same function.
(defn train-document
  ([classifier klass features]
   (multinomial/train-document classifier klass features 1))
  ([classifier klass features occ]
   (multinomial/train-document classifier klass features occ)))

(defn- condprob
  ([classifier c]
   (/ 1 (+ (common/Nc classifier c))))
  ([classifier t c]
   (/ (inc (common/Tct classifier t c))
      (+ (common/Nc classifier c)))))

(defn- score
  [classifier common-tokens uncommon-tokens klass]
  ;;TODO flat prior option!
  (+ (Math/log (common/prior classifier klass))
     (reduce + (map #(Math/log (condprob classifier % klass)) common-tokens))
     (reduce + (map #(Math/log (- 1 (condprob classifier % klass))) uncommon-tokens))))

(defn- apply-nb
  [classifier document]
  (let [classes (keys @(:classes classifier))
        all-tokens (keys @(:tokens classifier))
        tokens (preprocess-document document)
        common-tokens (filter (set tokens) all-tokens)
        uncommon-tokens (remove (set tokens) all-tokens)]
    (reduce into {} (pmap #(hash-map % (score classifier common-tokens uncommon-tokens %)) classes))))

(defrecord Bernoulli [all classes algorithm tokens score]
  NaiveBayes

  (train [classifier documents options]
    (doseq [doc documents]
      ;; TODO: Fix this ugly thing
      (if (= (count doc) 2)
        (let [[d c] doc
              f (preprocess-document d)]
          (train-document classifier c f))
        (let [[d c o] doc
              f (preprocess-document d)]
          (train-document classifier c f (Integer/parseInt o))))))

  (classify [classifier document n]
    (take n (sort-by val > (apply-nb classifier document))))

  (export [classifier]
    {:terms (for [[t cats] @(:tokens classifier)
                  [cid _] cats
                  :when (not (= :all cid))]
              [t cid (Math/log (condprob classifier t cid))])
     :cats (map (fn [c] [c
                        (Math/log (common/prior classifier c))
                        (Math/log (condprob classifier c))])
                (keys @(:classes classifier)))}))

(defn make-bernoulli []
  (map->Bernoulli {:all (atom {:n 0 :v 0})
                   :classes (atom {})
                   :algorithm {:name :bernoulli}
                   :tokens (atom {})
                   :score (atom :naive-bayes)}))
