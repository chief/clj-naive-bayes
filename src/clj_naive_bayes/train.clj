(ns clj_naive_bayes.train
  (:use [clj_naive_bayes.core]
        [clj_naive_bayes.utils]
        [clojure.java.io :only (reader)])
  (:require  [clojure.data.csv :as csv]))

(defn train-document!
  [classifier klass v]
  (swap! classifier update-in [:all :n] inc)

  (if (get-in @classifier [:classes klass])
    (swap! classifier update-in [:classes klass :n] inc)
    (do
      (swap! classifier assoc-in [:classes klass :n] 1)
      (swap! classifier assoc-in [:classes klass :st] 0)))


  (doseq [token v]
    (if (get-in @classifier [:all :tokens token])
      (do
        (swap! classifier update-in [:all :tokens token] inc)
        (swap! classifier update-in [:all :st] inc)
        (swap! classifier update-in [:classes klass :st] inc))
      (do
        (swap! classifier update-in [:all :v] inc)
        (swap! classifier assoc-in  [:all :tokens token] 1)))

    (if (get-in @classifier [:classes klass :tokens token])
      (swap! classifier update-in [:classes klass :tokens token] inc)
      (swap! classifier assoc-in  [:classes klass :tokens token] 1)) ))

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
          algorithm (@classifier :algorithm)
          v (flatten (process-features (features document (:fn options)) algorithm))]
      (train-document! classifier klass v))))

(defn train-from
  [classifier filename {:keys [limit]
                        :or {limit 100}}]
  (let [with-documents (take limit (load-data filename))]
    (train classifier with-documents)))

(defn parallel-train-from
  [classifier filename & {:keys [limit train-options]
                          :or {limit 100
                               train-options {:fn (partial take 2)}}}]
  (with-open [in-file (reader filename)]
    (let [with-documents (take limit (csv/read-csv in-file))]
      (dorun
        (pmap #(train classifier [%] :options train-options) with-documents)))))
