(ns clj-naive-bayes.train
  (:use [clojure.java.io :only (reader)])
  (:require [clj-naive-bayes.utils :as utils]
            [clojure.data.csv :as csv]
            [schema.core :as s]))

(defn train-class
  [classes klass]
  (if (get-in @classes [klass])
    (swap! classes update-in [klass :n] inc)
    (do
      (swap! classes assoc-in [klass :n] 1)
      (swap! classes assoc-in [klass :st] 0))))

(defmulti train-document
  "Trains classifier with document"
  (fn [classifier klass document] (get-in classifier [:algorithm :name])))

(defmethod train-document :default
  [classifier klass v]
  (let [all (:all classifier)
        classes (:classes classifier)
        tokens (:tokens classifier)]
    (swap! all update-in [:n] inc)

    (if (nil? (get @all :st))
      (swap! all assoc-in [:st] 0))

    (train-class classes klass)

    (doseq [token v]
      (if (get @tokens token)
        (do
          (swap! tokens update-in [token :all] inc)
          (swap! all update-in [:st] inc)
          (swap! classes update-in [klass :st] inc))
        (do
          (swap! all update-in [:st] inc)
          (swap! classes update-in [klass :st] inc)
          (swap! all update-in [:v] inc)
          (swap! tokens assoc-in  [token :all] 1)))

      (if (get-in @tokens [token klass])
        (swap! tokens update-in [token klass] inc)
        (swap! tokens assoc-in [token  klass] 1)))))

;; (defmethod train-document :multinomial-positional-nb
;;   [classifier klass v]
;;   (let [all (:all classifier)
;;         classes (:classes classifier)
;;         tokens (:tokens classifier)]

;;     (swap! all update-in [:n] inc)

;;     (if (get-in @classes [klass])
;;       (swap! classes update-in [klass :n] inc)
;;       (swap! classes assoc-in [klass :n] 1))

;;     (doseq [[token position] v]
;;       (if (nil? (get @tokens token))
;;         (swap! all update-in [:v] inc))

;;       (if (get-in @tokens [token :all position])
;;         (swap! tokens update-in [token :all position] inc)
;;         (swap! tokens assoc-in [token :all position] 1))

;;       (if (get-in @all [:st position])
;;         (swap! all update-in [:st position] inc)
;;         (swap! all assoc-in [:st position] 1))

;;       (if (get-in @classes [klass :st position])
;;         (swap! classes update-in [klass :st position] inc)
;;         (swap! classes assoc-in [klass :st position] 1))

;;       (if (get-in @tokens [token klass position])
;;         (swap! tokens update-in [token klass position] inc)
;;         (swap! tokens assoc-in [token klass position] 1)))))

(defn target
  "Returns a target from a trained document. This documents should have its
   target class at the end"
  [document]
  (peek document))

(defn features
  "Gets all features except class (last element)."
  [document]
  (pop document))

(defn train
  [classifier with-documents]
  (doseq [document with-documents]
    (let [klass (target document)
          algorithm (:algorithm classifier)
          v (first (utils/process-features classifier (features document)))]
      (train-document classifier klass v))))


;; (defn parallel-train-from
;;   [classifier filename & {:keys [limit train-options]
;;                           :or {limit 100}}]
;;   (with-open [in-file (reader filename)]
;;     (let [with-documents (take limit (csv/read-csv in-file))]
;;       (dorun
;;        (pmap #(train classifier [%]) with-documents)))))
