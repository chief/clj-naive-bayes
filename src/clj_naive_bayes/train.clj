(ns clj_naive_bayes.train
  (:use [clj_naive_bayes.core]
        [clojure.java.io :only (reader)])
  (:require [clj_naive_bayes.utils :as utils]
            [clojure.data.csv :as csv]
            [schema.core :as s]))

(defn- train-class
  ([classes klass]
   (train-class classes klass 1))
  ([classes klass occ]
   (if (get-in @classes [klass])
     (swap! classes update-in [klass :n] inc)
     (do
       (swap! classes assoc-in [klass :n] 1)
       (swap! classes assoc-in [klass :st] 0)))))

(defmulti train-document
  "Trains classifier with document"
  (fn [classifier & args] (get-in classifier [:algorithm :name])))

(defmethod train-document :default
  ([classifier klass v]
   (train-document classifier klass v 1))
  ([classifier klass v occ]
   (let [{:keys [all classes tokens]} classifier
         inc-occ #(+ % occ)]

     (swap! all update-in [:n] inc-occ)

     (if (nil? (get @all :st))
       (swap! all assoc-in [:st] 0))

     ;; TODO: fixme
     (train-class classes klass occ)

     (doseq [token v]
       (if (get @tokens token)
         (do
           (swap! tokens update-in [token :all] inc-occ)
           (swap! all update-in [:st] inc-occ)
           (swap! classes update-in [klass :st] inc-occ))
         (do
           (swap! all update-in [:st] inc-occ)
           (swap! classes update-in [klass :st] inc-occ)
           (swap! all update-in [:v] inc-occ)
           (swap! tokens assoc-in  [token :all] occ)))

       (if (get-in @tokens [token klass])
         (swap! tokens update-in [token klass] inc-occ)
         (swap! tokens assoc-in [token  klass] occ))))))

(defmethod train-document :multinomial-positional-nb
  [classifier klass v]
  (let [all (:all classifier)
        classes (:classes classifier)
        tokens (:tokens classifier)]

    (swap! all update-in [:n] inc)

    (if (get-in @classes [klass])
      (swap! classes update-in [klass :n] inc)
      (swap! classes assoc-in [klass :n] 1))

    (doseq [[token position] v]
      (if (nil? (get @tokens token))
        (swap! all update-in [:v] inc))

      (if (get-in @tokens [token :all position])
        (swap! tokens update-in [token :all position] inc)
        (swap! tokens assoc-in [token :all position] 1))

      (if (get-in @all [:st position])
        (swap! all update-in [:st position] inc)
        (swap! all assoc-in [:st position] 1))

      (if (get-in @classes [klass :st position])
        (swap! classes update-in [klass :st position] inc)
        (swap! classes assoc-in [klass :st position] 1))

      (if (get-in @tokens [token klass position])
        (swap! tokens update-in [token klass position] inc)
        (swap! tokens assoc-in [token klass position] 1)))))

(defn train-with-count [classifier documents]
  (doseq [document documents]
    (let [[d c o] document
          features (utils/process-features classifier d)]
      (train-document classifier c features o))))

(defn train-single [classifier documents]
  (doseq [document documents]
    (let [[d c] document
          algorithm (:algorithm classifier)
          features (utils/process-features classifier d)]
      (train-document classifier c features))))

(defn train
  [classifier documents]
  ;; Sample the first document to infer number of fields in each row.
  ;; TODO: Maybe use a multimethod here.
  (let [field-count (count (first documents))]
    (if (= field-count 3)
      (train-with-count classifier documents)
      (train-single classifier documents))))

(defn parallel-train
  ([classifier data]
   (parallel-train classifier data {}))
  ([classifier data {:keys [limit train-options] :or {limit 100}}]
   (let [data (take limit data)]
     (dorun
      (pmap #(train classifier [%]) data)))))

(defn parallel-train-from-file
  [classifier filename options]
  (let [data (utils/load-data-from-file filename)]
    (parallel-train classifier data options)))
