(ns clj_naive_bayes.train-test
  (:require [clj_naive_bayes.train :as train]
            [clj_naive_bayes.core :as core]
            [clojure.test :refer :all]))

(def document ["iphone 6" "mobile phones" "" "40"])

(deftest test-target
  (is (= (train/target document) "40")))

(deftest test-features
  (is (= (train/features document) ["iphone 6" "mobile phones" ""]))
  (is (= (train/features document (partial take 1)) ["iphone 6"])))

(deftest test-train
  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier)
        do-train (train/train classifier documents)
        all (:all classifier)
        tokens (:tokens classifier)
        classes (:classes classifier)]
    (is (= (get-in @all [:n]) 4))
    (is (= (get-in @all [:v]) 6))
    (is (= (get-in @all [:st]) 11))
    (is (= (get-in @tokens ["Chinese" :all]) 6))
    (is (= ["Chinese" "Beijing" "Shanghai" "Macao" "Tokyo" "Japan"] (keys @tokens)))
    (is (= (get-in @classes ["China" :n]) 3))
    (is (= (get-in @tokens ["Chinese" "China"]) 5)))

  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        classifier (core/new-classifier {:name :binary-nb})
        do-train (train/train classifier documents)
        all (:all classifier)
        tokens (:tokens classifier)
        lexicon (:lexicon classifier)
        classes (:classes classifier)]

    (is (= (get-in @all [:n]) 4))
    (is (= (get-in @all [:v]) 6))
    (is (= (get-in @tokens ["Chinese" :all]) 4))

    (is (= (get-in @classes ["China" :n]) 3))
    (is (= (get-in @tokens ["Chinese" "China"]) 3))))
