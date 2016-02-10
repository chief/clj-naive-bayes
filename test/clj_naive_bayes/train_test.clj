(ns clj-naive-bayes.train-test
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
        multinomial-classifier (core/new-classifier)
        do-train (train/train multinomial-classifier documents)
        all (:all multinomial-classifier)
        classes (:classes multinomial-classifier)]
    (is (= (get-in @all [:n]) 4))
    (is (= (get-in @all [:v]) 6))
    (is (= (get-in @all [:st]) 11))
    (is (= (get-in @all [:tokens "Chinese"]) 6))

    (is (= (get-in @classes ["China" :n]) 3))
    (is (= (get-in @classes ["China" :tokens "Chinese"]) 5)))

  (let [documents [["Chinese Beijing Chinese" "China"]
                   ["Chinese Chinese Shanghai" "China"]
                   ["Chinese Macao" "China"]
                   ["Tokyo Japan Chinese" "Japan"]]
        binary-classifier (core/new-classifier {:name :binary-nb})
        do-train (train/train binary-classifier documents)
        all (:all binary-classifier)
        classes (:classes binary-classifier)]

    (is (= (get-in @all [:n]) 4))
    (is (= (get-in @all [:v]) 6))
    (is (= (get-in @all [:tokens "Chinese"]) 4))

    (is (= (get-in @classes ["China" :n]) 3))
    (is (= (get-in @classes ["China" :tokens "Chinese"]) 3))))
