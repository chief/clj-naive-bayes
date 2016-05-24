(ns clj-naive-bayes.train-test
  (:require [clj-naive-bayes.train :as train]
            [clj-naive-bayes.core :as core]
            [clojure.test :refer :all]))

;; (def document ["iphone 6" "mobile phones" "" "40"])

;; (deftest test-target
;;   (is (= (train/target document) "40")))

;; (deftest test-features
;;   (is (= (train/features document) ["iphone 6" "mobile phones" ""])))

;; (deftest test-train
;;   (let [documents [["Chinese Beijing Chinese" "China"]
;;                    ["Chinese Chinese Shanghai" "China"]
;;                    ["Chinese Macao" "China"]
;;                    ["Tokyo Japan Chinese" "Japan"]]
;;         classifier (core/new-classifier)
;;         do-train (train/train classifier documents)
;;         all (:all classifier)
;;         tokens (:tokens classifier)
;;         classes (:classes classifier)]
;;     (is (= (get-in @all [:n]) 4))
;;     (is (= (get-in @all [:v]) 6))
;;     (is (= (get-in @all [:st]) 11))
;;     (is (= (get-in @tokens ["Chinese" :all]) 6))
;;     (is (= ["Chinese" "Beijing" "Shanghai" "Macao" "Tokyo" "Japan"] (keys @tokens)))
;;     (is (= (get-in @classes ["China" :n]) 3))
;;     (is (= (get-in @tokens ["Chinese" "China"]) 5)))

;;   (let [documents [["Chinese Beijing Chinese" "China"]
;;                    ["Chinese Chinese Shanghai" "China"]
;;                    ["Chinese Macao" "China"]
;;                    ["Tokyo Japan Chinese" "Japan"]]
;;         classifier (core/new-classifier {:name :multinomial-positional-nb})
;;         do-train (train/train classifier documents)
;;         all (:all classifier)
;;         tokens (:tokens classifier)
;;         classes (:classes classifier)]
;;     (is (= (get-in @all [:n]) 4))
;;     (is (= (get-in @all [:v]) 6))
;;     (is (= 4 (get-in @all [:st 0])))
;;     (is (= (get-in @tokens ["Chinese" :all 0]) 3))
;;     (is (= ["Chinese" "Beijing" "Shanghai" "Macao" "Tokyo" "Japan"] (keys @tokens)))
;;     (is (= (get-in @classes ["China" :n]) 3))
;;     (is (= (get-in @tokens ["Chinese" "China" 1]) 1)))

;;   (let [documents [["Chinese Beijing Chinese" "China"]
;;                    ["Chinese Chinese Shanghai" "China"]
;;                    ["Chinese Macao" "China"]
;;                    ["Tokyo Japan Chinese" "Japan"]]
;;         classifier (core/new-classifier {:name :binary-nb})
;;         do-train (train/train classifier documents)
;;         all (:all classifier)
;;         tokens (:tokens classifier)
;;         lexicon (:lexicon classifier)
;;         classes (:classes classifier)]

;;     (is (= (get-in @all [:n]) 4))
;;     (is (= (get-in @all [:v]) 6))
;;     (is (= (get-in @tokens ["Chinese" :all]) 4))

;;     (is (= (get-in @classes ["China" :n]) 3))
;;     (is (= (get-in @tokens ["Chinese" "China"]) 3))))
