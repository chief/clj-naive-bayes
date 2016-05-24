(ns clj-naive-bayes.classifiers.bernoulli-matrix
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as r]))

(m/set-current-implementation :vectorz)

(defn label-binarize [y classes]
  (let [sorted_class (sort classes)
        indices (map #(.indexOf sorted_class %) y)
        Y (m/new-sparse-array [(count y) (count classes)])]
    (doseq [i (range (count indices))]
      (m/mset! Y i (nth indices i) 1))
    Y))

(defn- fit-transform
  "Fit label encoder and return encoded labels."
  [y]
  ;; TODO: scikit makes certain the array is 1d here, let's skip it for now.
  (let [classes (distinct y)
        Y (label-binarize y classes)]
    [Y classes]))

(defn- _count
  "Count feature occurences."
  [X Y]
  ;; TODO: scikit makes certain data are binary here, let's skip it for now.
  (let [feature_count_ (m/inner-product (m/transpose Y) X)
        class_count_ (reduce m/add Y)]
    [feature_count_ class_count_]))

(defn- _update_feature_log_prob
  "Apply smoothing to raw counts and recompute log probabilities."
  [feature_count class_count alpha]
  (let [smoothed_fc (m/add feature_count alpha)
        smoothed_cc (m/add class_count (* alpha 2))]
    ;; TODO: This seems suboptimal, however the scikit's way of doing it
    ;; (using reshape) doesn't seem to work.
    (m/transpose (m/sub (m/log (m/transpose smoothed_fc)) (m/log smoothed_cc)))))

(defn- _update_class_log_prior [class_count]
  (m/sub (m/log class_count) (m/log (m/ereduce m/add class_count))))

(defn- _joint_log_likelihood [classifier X]
  (let [{:keys [feature-log-prob class-log-prior]} classifier
        [n-classes n-features] (m/shape feature-log-prob)
        [n-samples n-features-X] (m/shape X)
        neg-prob (m/log (m/sub 1 (m/exp feature-log-prob)))]
    ;; TODO: Check that n-features == n-features-X
    (m/add!
     (m/mmul X (m/transpose (m/sub feature-log-prob neg-prob)))
     class-log-prior
     (reduce m/add (m/transpose neg-prob)))))

;; TODO: Only this left
;; ...and validation
;; ...and error checking
;; But really, only this left!
;; return self.classes_[np.argmax(jll, axis=1)]
(defn predict [classifier X]
  (let [jll (-> (_joint_log_likelihood classifier X)
                m/to-nested-vectors)
        best (map #(.indexOf % (apply max %)) jll)]
    (map #(nth (:classes classifier) %) best)))

(defn fit
  "X: {array-like, sparse_matrix}, shape = [n_samples, n_features]
      Training vectors, where n_samples is the number of samples and
      n_features is the number of features.

   y: vector, shape = [n_samples]
      Target values."
  [X y alpha]
  ;; TODO: scikit performs some validation here on sizes of the matrices. Let's skip
  ;; it for now.
  (let [[_ n_features] (m/shape X)
        [Y classes] (fit-transform y)
        n_effective_classes (second (m/shape Y))
        ;;class_count_ (m/zero-vector n_effective_classes)
        ;;feature_count_ (m/zero-matrix n_effective_classes n_features)
        [feature_count class_count] (_count X Y)]
    {:feature-log-prob (_update_feature_log_prob feature_count class_count alpha)
     :class-log-prior (_update_class_log_prior class_count)
     :classes classes}))

;; TODO: classes should be sorted already!
(defn example []
  ;; TODO: make sure X can be sparse
  (let [X (r/sample-rand-int [6 100] 2)  ;; 6 documents, 100 terms
        Y [1 2 3 4 4 5]
        classifier (fit X Y 1.0)
        target (m/submatrix X [[2 1]])] ;; target classes
    (predict classifier target)))
