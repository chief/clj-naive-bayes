(ns clj_naive_bayes.train
  (:require [clj_naive_bayes.utils :as utils]
            [clojure.data.csv :as csv]
            [schema.core :as s]))

(defn- update-classes-multinomial
  ([classes klass]
   (update-classes-multinomial classes klass 1))
  ([classes klass occ]
   (if (get classes klass)
     (update-in classes [klass :n] inc)
     (-> (assoc-in classes [klass :n] 1)
         (assoc-in [klass :st] 0)))))

(defn- update-all-for-token-multinomial [all tokens token f]
  (if (get tokens token)
    (update all :st f)
    (-> (update all :v inc)
        (update :st f))))

(defn- update-all-for-token-multinomial-positional [all tokens token position f]
  (if (get tokens token)
    (update-in all [:st position] f)
    (-> (update all :v inc)
        (update-in [:st position] f))))

(defmulti train-document
  "Trains classifier with document"
  (fn [data algorithm & args] (:name algorithm)))

(defmethod train-document :default
  ([data algorithm klass features]
   (train-document data algorithm klass features 1))
  ([data algorithm klass features occ]
   (let [inc-occ #(+ (or % 0) occ)
         pre (update data :classes update-classes-multinomial klass occ)
         d (reduce (fn [{:keys [all classes tokens]} token]
                     {:all (update-all-for-token-multinomial all tokens token inc-occ)
                      :tokens (-> (update-in tokens [token :all] inc-occ)
                                  (update-in [token klass] inc-occ))
                      :classes (update-in classes [klass :st] inc-occ)})
                   pre features)]
     (-> (update-in d [:all :st] #(or % 0))
         (update-in [:all :n] inc-occ)))))

(defmethod train-document :multinomial-positional-nb
  ([data algorithm klass features]
   (train-document data algorithm klass features 1))
  ([data algorithm klass features occ]
   (let [inc-occ #(+ (or % 0) occ)
         pre (-> (update-in data [:classes klass :n] inc-occ)
                 (update-in [:all :n] inc))]
     (reduce (fn [{:keys [all classes tokens]} [token position]]
               {:all (update-all-for-token-multinomial-positional all tokens token position inc-occ)
                :tokens (-> (update-in tokens [token :all position] inc-occ)
                            (update-in [token klass position] inc-occ))
                :classes (update-in classes [klass :st position] inc-occ)})
             pre features))))

(defn train-with-count [classifier documents]
  (doseq [document documents]
    (let [[d c o] document
          algorithm (:algorithm classifier)
          features (utils/process-features classifier d)]
      (swap! (:data classifier) train-document algorithm c features o))))

(defn train-single [classifier documents]
  (doseq [document documents]
    (let [[d c] document
          algorithm (:algorithm classifier)
          features (utils/process-features classifier d)]
      (swap! (:data classifier) train-document algorithm c features))))

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
  ([classifier data {:keys [limit train-options] :or {limit (count data)}}]
   (let [data (take limit data)]
     (dorun
      (pmap #(train classifier [%]) data)))))

(defn parallel-train-from-file
  [classifier filename options]
  (let [data (utils/load-data-from-file filename)]
    (parallel-train classifier data options)))
