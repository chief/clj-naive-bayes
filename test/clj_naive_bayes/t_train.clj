(ns clj_naive_bayes.t-train
  (:use midje.sweet)
  (:use [clj_naive_bayes.train]
        [clj_naive_bayes.core]))

(def document ["iphone 6" "mobile phones" "" "40"])

(facts "about `target`"
  (fact "it returns the correct target"
    (target document) => "40"))

(facts "about `features`"
  (fact "it returns all features if nothing is passed"
    (features document) => ["iphone 6", "mobile phones", ""])
  (fact "it returns features based on f"
    (features document (partial take 1)) => ["iphone 6"]))

(facts "about `train`"
  (fact "it trains correctly when algorithm is :multinomial-nb"
    (let [documents [["Chinese Beijing Chinese" "China"]
                     ["Chinese Chinese Shanghai" "China"]
                     ["Chinese Macao" "China"]
                     ["Tokyo Japan Chinese" "Japan"]]
          multinomial-classifier (clj_naive_bayes.core/new-classifier)]

      (train multinomial-classifier documents)

      (get-in @multinomial-classifier [:all :n]) => 4
      (get-in @multinomial-classifier [:all :v]) => 6
      (get-in @multinomial-classifier [:all :tokens "Chinese"]) => 6

      ; :classes aggregations
      (get-in @multinomial-classifier [:classes "China" :n]) => 3
      (get-in @multinomial-classifier [:classes "China" :tokens "Chinese"]) => 5))

   (fact "it trains correctly when algorithm is :binary-nb"
    (let [documents [["Chinese Beijing Chinese" "China"]
                     ["Chinese Chinese Shanghai" "China"]
                     ["Chinese Macao" "China"]
                     ["Tokyo Japan Chinese" "Japan"]]
          binary-classifier (clj_naive_bayes.core/new-classifier :binary-nb)]

      (train binary-classifier documents)

      (get-in @binary-classifier [:all :n]) => 4
      (get-in @binary-classifier [:all :v]) => 6
      (get-in @binary-classifier [:all :tokens "Chinese"]) => 4

      ; :classes aggregations
      (get-in @binary-classifier [:classes "China" :n]) => 3
      (get-in @binary-classifier [:classes "China" :tokens "Chinese"]) => 3)))