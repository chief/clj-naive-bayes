(ns clj-naive-bayes.classifiers.multinomial
  (:require [clj-naive-bayes.protocols :as p]
            [clj-naive-bayes.utils :as utils]
            [clj-naive-bayes.classifiers.common :as common]
            [schema.core :as s]))

(defn- preprocess-document [document]
  (utils/tokenize document))

(defn train-document
  ([classifier klass features]
   (train-document classifier klass features 1))
  ([classifier klass features occ]
   (let [{:keys [all classes tokens]} classifier
         inc-occ #(+ % occ)]
     (swap! all update :n inc-occ)

     (if (nil? (get @all :st))
       (swap! all assoc :st 0))

     (if (get-in @classes [klass])
       (swap! classes update-in [klass :n] inc-occ)
       (do
         (swap! classes assoc-in [klass :n] occ)
         (swap! classes assoc-in [klass :st] 0)))

     (doseq [feature features]
       (if (get @tokens feature)
         (do
           (swap! tokens update-in [feature :all] inc-occ)
           (swap! all update :st inc-occ)
           (swap! classes update-in [klass :st] inc-occ))
         (do
           (swap! all update :st inc-occ)
           (swap! classes update-in [klass :st] inc-occ)
           (swap! all update :v inc-occ)
           (swap! tokens assoc-in [feature :all] occ)))

       (if (get-in @tokens [feature klass])
         (swap! tokens update-in [feature klass] inc-occ)
         (swap! tokens assoc-in [feature klass] occ))))))

(defn- condprob
  ([classifier c]
   (/ 1
      (+ (common/NTct classifier c) (common/B classifier))))
  ([classifier t c]
   (/ (inc (common/Tct classifier t c))
      (+ (common/NTct classifier c) (common/B classifier)))))

(defn- score
  [classifier tokens klass]
  ;;TODO flat prior option!
  (+ (Math/log (common/prior classifier klass))
     (reduce + (map #(Math/log (condprob classifier % klass)) tokens))))

(defn- apply-nb
  [classifier document]
  (let [classes (keys @(:classes classifier))
        tokens (preprocess-document document)]
    (reduce into {} (map #(hash-map % (score classifier tokens %)) classes))))

(defrecord Multinomial [all classes algorithm tokens score]
  p/NaiveBayes

  ;; TODO: This probably should/can use pmap
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
    (sort-by val > (apply-nb document)))

  (export [classifier]
    {:terms (for [[t cats] @(:tokens classifier)
                  [cid _] cats
                  :when (not (= :all cid))]
              [t cid (Math/log (condprob classifier t cid))])
     :cats (map (fn [c] [c
                        (Math/log (common/prior classifier c))
                        (Math/log (condprob classifier c))])
                (keys @(:classes classifier)))}))

(defn make-multinomial []
  (map->Multinomial {:all (atom {:n 0 :v 0})
                     :classes (atom {})
                     :algorithm {:name :multinomial-nb
                                 :ngram-type :multinomial}
                     :tokens (atom {})
                     :score (atom :naive-bayes)}))
