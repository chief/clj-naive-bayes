(ns clj_naive_bayes.t-utils
  (:use midje.sweet)
  (:use [clj_naive_bayes.utils]))

(facts "about `tokenize`"
  (fact "it tokenizes strings correctly"
    (tokenize "this is a test") => ["this" "is" "a" "test"]))

(facts "about `ngram-keys`"
  (fact "it returns ngram keys"
    (ngram-keys ["iphone" "6" "32"]) => ["iphone_6" "6_32"]))

(facts "about `process`"
  (fact "it returns processed features when algorithm is :multinomial-nb"
    (process-features ["iphone 6" "mobile phones"] {:name :multinomial-nb}) =>
      [["iphone" "6"] ["mobile" "phones"]])

  (fact "it returns processed features when algorithm is :binary-nb"
    (process-features ["iphone 6 6" "mobile phones"] {:name :binary-nb}) =>
      [["iphone" "6"] ["mobile" "phones"]])

  (fact "it returns processed features when algorithm is :ngram-nb"
    (process-features ["iphone 6 32" "mobile phones κινητα"] {:name :ngram-nb}) =>
      [["iphone_6" "6_32"] ["mobile_phones" "phones_κινητα"]]))