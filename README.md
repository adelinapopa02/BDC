# BDC — Big Data Computing Homeworks

Apache Spark implementations for the Big Data Computing course (University of Padua). Two homeworks: fair k-center clustering on batch data, and frequent-items mining on a data stream.

## Structure

```
src/
  G70HW1.java           Fair Farthest-First Traversal clustering (Spark Core, batch)
  G70HW2.java            Frequent items on a stream: Sticky Sampling + Count-Min Sketch (Spark Streaming)
  WordCountExample.java  Course-provided Spark word-count example
build.gradle             Gradle build (Java, Spark Core 3.5.1 + Spark MLlib 3.5.1)
testinputN32D2.csv, testinputN72D4.csv   Sample point sets for HW1
uber_small.csv, sentence_small.txt        Additional sample datasets
H2_runs.txt              Sample stdout log from an HW2 run
```

## Build

```bash
./gradlew build
```

Produces `build/libs/BDC.jar`.

## HW1 — Fair k-center Clustering (`G70HW1`)

Points are labeled `A` or `B` (two demographic groups). The program computes `kA + kB` centers minimizing the maximum distance from any point to its nearest center, under per-group budgets `kA` and `kB`.

- **`FairFFT`** — sequential Fair Farthest-First Traversal: greedily picks the point farthest from the current center set, respecting the remaining per-group budget, until `kA + kB` centers are chosen.
- **`MRFairFFT`** — 2-round MapReduce-style distributed version: round 1 runs `FairFFT` independently on each of the `L` partitions to build a coreset; round 2 runs `FairFFT` again on the collected coreset to produce the final solution.
- **`computeObjective`** — evaluates the clustering radius (max point-to-nearest-center distance) over the full dataset.

Run with `spark-submit`:

```bash
spark-submit --class G70HW1 build/libs/BDC.jar <file_path> <kA> <kB> <L>
```

- `file_path` — CSV file, each line `x_1,...,x_D,label` (label is `A` or `B`)
- `kA`, `kB` — center budgets for groups A and B
- `L` — number of partitions (coreset granularity)

Example: `spark-submit --class G70HW1 build/libs/BDC.jar testinputN32D2.csv 2 2 4`

## HW2 — Frequent Items on a Stream (`G70HW2`)

Reads `n` integer items from a socket stream and estimates which items have relative frequency ≥ `phi`, comparing two sketching algorithms against the ground truth:

- **Sticky Sampling** — reservoir-style sampling with retention probability `r/n` (`r = ln(1/(δφ)) / ε`); reports items with sampled count ≥ `(φ - ε)n`.
- **Count-Min Sketch** — `d × w` hash table of pairwise-independent hash functions; reports items whose estimated count ≥ `φn`.

Both are compared against exact frequencies computed by counting every item in the stream.

Run with `spark-submit` against a socket stream (course server `algo.dei.unipd.it`):

```bash
spark-submit --class G70HW2 build/libs/BDC.jar <n> <phi> <epsilon> <delta> <d> <w> <portExp>
```

- `n` — number of stream items to process
- `phi` — frequency threshold
- `epsilon`, `delta` — Sticky Sampling accuracy/confidence parameters
- `d`, `w` — Count-Min Sketch depth and width
- `portExp` — port to read the stream from

Example output of a run is in `H2_runs.txt`.

## Deliverable forms

`src/G70HW2form.pdf` / `src/G70HW2/G70HW2form.pdf` are the course's homework report forms.
