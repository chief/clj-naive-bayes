(ns clj_naive_bayes.core-test
  (:require [clj_naive_bayes.core :as core]
            [clj_naive_bayes.train :as train]
            [clojure.test :refer :all]))

(deftest test-classify
  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        multinomial-classifier (core/new-classifier)]
    (train/train multinomial-classifier documents)

    (is (= (core/classify multinomial-classifier ["Chinese"]) "China"))
    (is (= (core/classify multinomial-classifier []) "China"))))
