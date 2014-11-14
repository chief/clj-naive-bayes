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
      [["iphone_6" "6_32"] ["mobile_phones" "phones_κινητα"]])

  (fact "it returns processed features when algorithm is :ngram-nb and ngram-size is 3"
    (process-features ["iphone 6 32" "mobile phones κινητα"]
                      {:name :ngram-nb :ngram-size 3}) =>
      [["iphone_6_32"] ["mobile_phones_κινητα"]])

  (fact "it returns processed features when algorithm is :ngram-nb ngram-size is 2
        and ngram-type is :multinomial"
    (process-features ["iphone 6 iphone 6" "mobile phones"]
                      {:name :ngram-nb :ngram-size 2 :ngram-type :multinomial}) =>
      [["iphone_6" "6_iphone" "iphone_6"] ["mobile_phones"]])

  (fact "it returns processed features when algorithm is :ngram-nb ngram-size is 2
        and ngram-type is :binary"
    (process-features ["iphone 6 iphone 6" "mobile phones"]
                      {:name :ngram-nb :ngram-size 2 :ngram-type :binary}) =>
      [["iphone_6" "6_iphone"] ["mobile_phones"]])

  (fact "it returns processed features when algorithm is :ngram-nb
        ngram-size is 2 ngram-type is :binary and :boost_start is `true`"
    (process-features ["iphone 6 iphone 6" "mobile phones"]
                      {:name :ngram-nb :ngram-size 2 :ngram-type :binary
                       :boost-start true}) =>
      [["_iphone" "iphone_6" "6_iphone"] ["_mobile" "mobile_phones"]]))


(facts "about `add-empty-space-before"
  (fact "it adds empty space before an array"
    (add-empty-space-before ["1" "2" "3"]) => ["" "1" "2" "3"]))