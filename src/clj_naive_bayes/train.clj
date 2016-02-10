(ns clj_naive_bayes.train
  (:use [clj_naive_bayes.core]
        [clj_naive_bayes.utils]
        [clojure.java.io :only (reader)])
  (:require  [clojure.data.csv :as csv]
             [schema.core :as s]))

(defmulti train-document
  "Trains classifier with document"
  (fn [classifier klass document] (get-in classifier [:algorithm :name])))

(defmethod train-document :default
  [classifier klass v]
  (let [all (:all classifier)
        classes (:classes classifier)]

    (swap! all update-in [:n] inc)

    (if (get-in @classes [klass])
      (swap! classes update-in [klass :n] inc)
    (do
      (swap! classes assoc-in [klass :n] 1)
      (swap! classes assoc-in [klass :st] 0)))

    (doseq [token v]
      (if (get-in @all [:tokens token])
        (do
          (swap! all update-in [:tokens token] inc)
          (swap! all update-in [:st] inc)
          (swap! classes update-in [klass :st] inc))
        (do
          (swap! all update-in [:st] inc)
          (swap! classes update-in [klass :st] inc)
          (swap! all update-in [:v] inc)
          (swap! all assoc-in  [:tokens token] 1)))

      (if (get-in @classes [klass :tokens token])
        (swap! classes update-in [klass :tokens token] inc)
        (swap! classes assoc-in  [klass :tokens token] 1)))))

(defn target
  "Returns a target from a trained document. This documents should have its
   target class at the end"
  [document]
  (last document))

(defn features
  "Selects features from the vector feature list based on partial function f or
   just returns them all."
  ([document]
   (pop document))
  ([document f]
   (f (pop document))))

(defn train
  [classifier with-documents & {:keys [options]
                                :or {options {:fn (partial take 3)}}}]
  (doseq [document with-documents]
    (let [klass (target document)
          algorithm (:algorithm classifier)
          v (flatten (process-features classifier (features document (:fn options))))]
      (train-document classifier klass v))))

(defn parallel-train-from
  [classifier filename & {:keys [limit train-options]
                          :or {limit 100
                               train-options {:fn (partial take 4)}}}]
  (with-open [in-file (reader filename)]
    (let [with-documents (take limit (csv/read-csv in-file))]
      (dorun
        (pmap #(train classifier [%] :options train-options) with-documents)))))
