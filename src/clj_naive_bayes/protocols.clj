(ns clj-naive-bayes.protocols)

(defprotocol NaiveBayes
  (train [classifier documents options])
  (classify [classifier document n])
  (export [classifier]))
