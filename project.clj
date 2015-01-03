(defproject clj-naive-bayes "0.0.1-SNAPSHOT"
  :description "Cool new project to do things and stuff"
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [cheshire "5.3.1"]
                 [org.clojure/core.memoize "0.5.6"]
                 [org.clojure/tools.namespace "0.2.7"]
                 [com.stuartsierra/component "0.2.2"]
                 [org.clojure/data.csv "0.1.2"]
                 [spyscope "0.1.5"]]
  :jvm-opts ["-Xmx4g"]
  :plugins [[lein-marginalia "0.8.0"]]
  :profiles {:dev {:dependencies [[midje "1.6.3"]]}})
