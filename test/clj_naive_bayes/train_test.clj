(ns clj_naive_bayes.train-test
  (:require [clj_naive_bayes.train :as train]
            [clj_naive_bayes.core :as core]
            [clojure.test :refer :all]))

(deftest test-train
  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier)
        do-train (train/train classifier documents)
        data @(:data classifier)
        all (:all data)
        tokens (:tokens data)
        classes (:classes data)]

    (is (= (:n all) 4))
    (is (= (:v all) 6))
    (is (= (:st all) 11))
    (is (= (get-in tokens ["chinese" :all]) 6))
    (is (= ["chinese" "beijing" "shanghai" "macao" "tokyo" "japan"] (keys tokens)))
    (is (= (get-in classes ["China" :n]) 3))
    (is (= (get-in tokens ["chinese" "China"]) 5)))

  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier {:name :multinomial-positional-nb})
        do-train (train/train classifier documents)
        data @(:data classifier)
        all (:all data)
        tokens (:tokens data)
        classes (:classes data)]

    (is (= (:n all) 4))
    (is (= (:v all) 6))
    (is (= (get-in all [:st 0]) 4))
    (is (= (get-in tokens ["chinese" :all 0]) 3))
    (is (= ["chinese" "beijing" "shanghai" "macao" "tokyo" "japan"] (keys tokens)))
    (is (= (get-in classes ["China" :n]) 3))
    (is (= (get-in tokens ["chinese" "China" 1]) 1)))

  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier {:name :binary-nb})
        do-train (train/train classifier documents)
        data @(:data classifier)
        all (:all data)
        tokens (:tokens data)
        lexicon (:lexicon data)
        classes (:classes data)]

    (is (= (:n all) 4))
    (is (= (:v all) 6))
    (is (= (get-in tokens ["chinese" :all]) 4))

    (is (= (get-in classes ["China" :n]) 3))
    (is (= (get-in tokens ["chinese" "China"]) 3))))
