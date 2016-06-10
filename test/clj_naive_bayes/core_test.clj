(ns clj_naive_bayes.core-test
  (:require [clj_naive_bayes.core :as core]
            [clj_naive_bayes.train :as train]
            [clojure.test :refer :all]))

(deftest test-classify-multinomial
  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier)]
    (train/train classifier documents)

    (is (= (core/condprob classifier "chinese" "China") 3/7))
    (is (= (core/condprob classifier "tokyo" "China") 1/14))
    (is (= (core/condprob classifier "japan" "China") 1/14))
    (is (= (core/condprob classifier "chinese" "Japan") 2/9))
    (is (= (core/condprob classifier "tokyo" "Japan") 2/9))
    (is (= (core/condprob classifier "japan" "Japan") 2/9))
    (is (= (core/B classifier) 6))
    (is (= (core/classify classifier ["Chinese"]) "China"))
    (is (= (core/classify classifier []) "China"))))

(deftest test-classify-bernoulli
  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier {:name :bernoulli} :default)]
    (train/train classifier documents)

    (is (= (core/condprob classifier "chinese" "China") 4/5))
    (is (= (core/condprob classifier "tokyo" "China") 1/5))
    (is (= (core/condprob classifier "japan" "China") 1/5))
    (is (= (core/condprob classifier "chinese" "Japan") 2/3))
    (is (= (core/condprob classifier "tokyo" "Japan") 2/3))
    (is (= (core/condprob classifier "japan" "Japan") 2/3))
    (is (= (core/classify classifier ["Chinese"]) "China"))
    (is (= (core/classify classifier []) "China"))))

(deftest test-naive-bayes
  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier {:name :multinomial-nb} :default)]
    (train/train classifier documents)

    (is (= (core/score classifier ["chinese"] "China") -1.1349799328389845))))
