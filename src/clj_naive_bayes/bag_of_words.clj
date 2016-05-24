(ns clj-naive-bayes.bag-of-words)

(defn create-hash [documents]
  (let [words (-> (apply into documents)
                  distinct)]
    (zipmap words (range))))

(defn create [documents]
  {:dict (create-hash documents)})
