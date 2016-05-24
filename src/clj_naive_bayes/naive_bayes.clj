(ns clj-naive-bayes.naive-bayes
  (:require [clj-naive-bayes.protocols :as p]
            [clj-naive-bayes.classifiers.multinomial :as multinomial]
            [clj-naive-bayes.classifiers.bernoulli :as bernoulli]))

;; Indirect to the protocol ns to create a simple API to the classifier.
;; TODO: Is there a better way to achieve this?
(defn train [classifier data options]
  (p/train classifier data options))

(defn classify
  [classifier document]
  (first (p/classify classifier document 1)))

;; (defn top-n-classes
;;   [classifier document n]
;;   (p/classify classifier document n))

(defn export [classifier]
  (p/export classifier))

(defn make-naive-bayes
  ([] (make-naive-bayes :multinomial))
  ([algorithm]
   (condp = algorithm
     :multinomial (multinomial/make-multinomial)
     :bernoulli  (bernoulli/make-bernoulli))))

