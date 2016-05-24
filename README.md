# clj-naive-bayes

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

## Training

Suppose you have a training dataset.

```clojure

(use 'clj_naive_bayes.train)

(parallel-train-from my-classifier "resources/train.csv" :limit 400000)

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
