(ns clj_naive_bayes.utils-test
  (:require [clj_naive_bayes.utils :as utils]
            [clojure.test :refer :all]))

(deftest test-tokenize
  (is (= (utils/tokenize "this is a test") ["this" "is" "a" "test"])))

(deftest test-ngram-keys
  (is (= (utils/ngram-keys ["iphone" "6" "32"]) ["iphone_6" "6_32"])))

(deftest test-process
  (is (= (utils/process-features ["iphone 6" "mobile phones"]
                                 {:name :multinomial-nb})
         [["iphone" "6"] ["mobile" "phones"]]))
  (is (= (utils/process-features ["Chinese" "mobile phones" "Chinese"]
                                 {:name :bernoulli})
         ["Chinese" "mobile" "phones"]))
  (is (= (utils/process-features ["iphone 6 6" "mobile phones"]
                                 {:name :binary-nb})
         [["iphone" "6"] ["mobile" "phones"]]))
  (is (= (utils/process-features ["iphone 6 32" "mobile phones κινητα"]
                                 {:name :ngram-nb})
         [["iphone_6" "6_32"] ["mobile_phones" "phones_κινητα"]]))
  (is (= (utils/process-features ["iphone 6 32" "mobile phones κινητα"]
                                 {:name :ngram-nb :ngram-size 3})
         [["iphone_6_32"] ["mobile_phones_κινητα"]]))
  (is (= (utils/process-features ["iphone 6 iphone 6" "mobile phones"]
                                 {:name :ngram-nb
                                  :ngram-size 2
                                  :ngram-type :multinomial})
         [["iphone_6" "6_iphone" "iphone_6"] ["mobile_phones"]]))
  (is (= (utils/process-features ["iphone 6 iphone 6" "mobile phones"]
                                 {:name :ngram-nb
                                  :ngram-size 2
                                  :ngram-type :binary})
         [["iphone_6" "6_iphone"] ["mobile_phones"]]))
  (is (= (utils/process-features ["iphone 6 iphone 6" "mobile phones"]
                                 {:name :ngram-nb
                                  :ngram-size 2
                                  :ngram-type :binary
                                  :boost-start true})
         [["_iphone" "iphone_6" "6_iphone"] ["_mobile" "mobile_phones"]]))
  (is (= (utils/process-features ["iphone 6 iphone 6" "mobile phones"]
                                 {:name :ngram-nb
                                  :ngram-size 2
                                  :ngram-type :binary
                                  :boost-start true
                                  :keep-sorted true})
         [["_iphone" "6_iphone"] ["_mobile" "mobile_phones"]])))

(deftest test-add-empty-space-before
  (is (= (utils/add-empty-space-before ["1" "2" "3"]) ["" "1" "2" "3"])))
