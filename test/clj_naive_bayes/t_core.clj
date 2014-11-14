(ns clj_naive_bayes.t-core
  (:use midje.sweet)
  (:use [clj_naive_bayes.core]
        [clj_naive_bayes.train]))

(facts "about `classify`"
  (fact "it classifies correctly when algorithm is :multinomial-nb"
    (let [documents [["Chinese Beijing Chinese" "China"]
                     ["Chinese Chinese Shanghai" "China"]
                     ["Chinese Macao" "China"]
                     ["Tokyo Japan Chinese" "Japan"]]
          multinomial-classifier (clj_naive_bayes.core/new-classifier)]

      (train multinomial-classifier documents)

      (classify multinomial-classifier ["Chinese"])) => "China"))