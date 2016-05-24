(ns clj-naive-bayes.utils-test
  (:require [clj-naive-bayes.utils :as utils]
            [clj-naive-bayes.core :as core]
            [clojure.test :refer :all]))

;; (deftest test-tokenize
;;   (is (= (utils/tokenize "this is a test") ["this" "is" "a" "test"])))

;; ;; (deftest test-ngram-keys
;; ;;   (is (= (utils/ngram-keys ["iphone" "6" "33"])
;; ;;          (lazy-seq (lazy-seq ["iphone_6" "6_33"]))))

;; ;;   (is (= (utils/ngram-keys ["iphone" "6" "32"] :explode-ngrams true)
;; ;;          (lazy-seq (lazy-seq ["iphone_6" "6_32"]) (lazy-seq ["iphone" "6" "32"])))))

;; (deftest test-process-features
;;   (is (= (utils/process-features (core/new-classifier) ["iphone 6" "mobile phones"])
;;          ["iphone" "6" "mobile" "phones"]))

;;   (is (= (utils/process-features (core/new-classifier {:name :bernoulli})
;;                                  ["Chinese" "mobile phones" "Chinese"])
;;          ["Chinese" "mobile" "phones"]))

;;   (is (= (utils/process-features (core/new-classifier {:name :binary-nb})
;;                                  ["iphone 6 6" "mobile phones"])
;;          ["iphone" "6" "mobile" "phones"]))

;;   (is (= (utils/process-features (core/new-classifier {:name :ngram-nb})
;;                                  ["iphone 6 32" "mobile phones κινητα"])
;;          [["iphone_6" "6_32"] ["mobile_phones" "phones_κινητα"]]))

;;   (is (= (utils/process-features (core/new-classifier {:name :ngram-nb
;;                                                        :ngram-size 3})
;;                                  ["iphone 6 32" "mobile phones κινητα"])
;;          [["iphone_6_32"] ["mobile_phones_κινητα"]]))

;;   (is (= (utils/process-features (core/new-classifier {:name :ngram-nb
;;                                                        :ngram-size 2

;;                                                        :ngram-type :multinomial})
;;                                  ["iphone 6 iphone 6" "mobile phones"])
;;          [["iphone_6" "6_iphone" "iphone_6"] ["mobile_phones"]]))

;;   (is (=  (first (utils/process-features
;;                   (core/new-classifier {:name :multinomial-positional-nb})
;;                   ["apple iphone 32 GB" "phones"]))
;;          ["apple" 0]))

;;   (is (= (utils/process-features (core/new-classifier {:name :ngram-nb
;;                                                        :ngram-size 2
;;                                                        :ngram-type :binary})
;;                                  ["iphone 6 iphone 6" "mobile phones"])
;;          [["iphone_6" "6_iphone"] ["mobile_phones"]])))
