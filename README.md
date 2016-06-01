# clj-naive-bayes

_Warning_: This project is under heavy development. Things will break!

## Usage

First of all you will need a new classifier:

```clojure

(require '[clj_naive_bayes.core :as nb])

(def my-classifier (nb/new-classifier {:name :ngram-nb :ngram-size 2 :ngram-type :multinomial}))

```

### Available options are

* __:name__ : Currently `:ngram-nb`, `:multinomial-nb` and `:binary-nb` are
  supported. (Default `:multinomial-nb`)

* __:ngram-size__ : Sets ngram size. (Default 2)

* __:ngram-type__ : Whether the ngram should be `:binary` or `:multinomial`

* __:boost-start__ : Boolean. (Default `false`). This flag has only effect
  with ngrams.

* __:keep-sorted__ : Boolean. (Default `false`). With this flag on all tokens
  in ngram keys are stores in alphabetical order.

## Train

Suppose you have a training dataset. This should be a CSV file, consisting of
lines with `<document,class>` or `<document,class,count>` elements. In the
second case, the `count` column should contain the number of occurences of each
sample. This is purely for space-saving purposes, so e.g. instead of using five
lines of the same `<document,class>` pair, a single `<document,class,5>` line
can be used instead.

```clojure

(require '[clj_naive_bayes.train :as train])

(train/parallel-train-from-file my-classifier "resources/train.csv" :limit 400000)

```

## Classify

Now we can try classifying a new document:

```clojure
(nb/classify my-classifier "iphone 6s")
=> "40"
```

## Export Probabilities to a Hashmap

This could be useful for e.g. persisting the classifier:

```clojure
(def out (nb/export a))
=> #'user/out
(keys out)
=> (:terms :cats)
```

## Evaluate Performance

```clojure

(use 'clj_naive_bayes.core)
(use 'clj_naive_bayes.eval)

(def logs (parallel-classifications my-classifier "resources/test.json"))

```

## Persist classifiers

Currently only file disk persistance is supported. Suppose you have a trained
classifier named `my-classifier` you can write it to a file:

```clojure

(use 'clj_naive_bayes.utils)

(persist-classifier my-classifier "resources/data.clj")

```

And later on load it:

```clojure

(use 'clj_naive_bayes.utils)

  (load-classifier my-classifier "resources/data.clj")

```

### Testing

`lein test` will run all tests.

`lein test [TEST]` will run only tests in the TESTS namespaces.

## Tooling

### Kibit

`lein kibit` will analyze code

### Marginalia

`lein marg` will produce documentation under `/docs`
