(defproject clj-naive-bayes "0.0.1-SNAPSHOT"
  :description "Cool new project to do things and stuff"
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [cheshire "5.5.0"]
                 [org.clojure/core.memoize "0.5.8"]
                 [org.clojure/tools.namespace "0.2.10"]
                 [com.stuartsierra/component "0.3.1"]
                 [org.clojure/data.csv "0.1.3"]
                 [spyscope "0.1.5"]]
  :jvm-opts ["-Xmx4g"]
  :plugins [[lein-marginalia "0.8.0"]]
  :profiles {:dev {:dependencies [[midje "1.8.2"]]}})
