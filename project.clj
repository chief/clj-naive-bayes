(defproject clj-naive-bayes "0.1.1"
  :description "Naive bayes in Clojure!"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [cheshire "5.5.0"]
                 [org.clojure/core.memoize "0.5.8"]
                 [org.clojure/tools.namespace "0.2.11"]
                 [com.stuartsierra/component "0.3.1"]
                 [org.clojure/data.csv "0.1.3"]
                 [spyscope "0.1.5"]
                 [prismatic/schema "1.0.4"]]
  :jvm-opts ["-Xmx4g"]
  :plugins [[lein-marginalia "0.8.0"]
            [lein-cljfmt "0.5.3"]])
