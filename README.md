# Authorship Attribution with Bidirectional LSTM

A PySpark and TensorFlow/Keras pipeline that predicts the author of an unseen 200-word passage based on writing style alone, built to power a fan-facing "guess the author" bot for a book press.

---

## Business Context

The press runs a public engagement programme built on interactive tools derived from its catalogue. This project adds authorship attribution as the next feature: a reader pastes a passage, the model returns a playful guess at which of the press's authors it most resembles.

The project answers two questions before deployment:

1. **Is there enough stylistic signal in the corpus for a model to distinguish the authors?**
2. **Are the model's errors explainable enough to be acceptable in a public-facing product?**

For full context, findings, and business recommendations, see the accompanying report.

---

## Authors in the Corpus

Eight authors from Project Gutenberg:

| Author | Era / Tradition |
|---|---|
| Arthur Conan Doyle | British, 19th–20th century |
| Bernard Shaw | Irish, 19th–20th century |
| Fyodor Dostoyevsky | Russian (English translation) |
| Herman Melville | American, 19th century |
| Jack London | American, 19th–20th century |
| Jonathan Swift | Anglo-Irish, 18th century |
| Leo Tolstoy | Russian (English translation) |
| Nathaniel Hawthorne | American, 19th century |

---

## Dataset

- **Source:** Project Gutenberg plain-text book files, organised as `dataset/<Author Name>/<Book Title>.txt`
- **Chunking:** 200-word non-overlapping chunks
- **Splits:** stratified by author, split by **book** (not by chunk) to prevent data leakage
  - Training: 32,613 chunks
  - Validation: 4,540 chunks
  - Test: 7,796 chunks

Splitting by book ensures no chunks from the same book appear in both training and test sets — a critical safeguard in authorship attribution, where a model could otherwise memorise book-specific phrases rather than learn an author's style.

---

## Pipeline Overview

### 1. Data Ingestion and Cleaning (`PySpark`)
- Read all `.txt` files recursively with `spark.read.text(..., recursiveFileLookup=True)`.
- Extract `author` from the folder name and `book_name` from the filename.
- Remove null/empty chunks, duplicates, symbol-heavy entries, and non-printable characters.
- Normalise whitespace.

### 2. Exploratory Data Analysis
- Author balance across chunks and books.
- Text length and vocabulary statistics per author.
- Word clouds for the top 200 words per author.
- PCA and t-SNE on document embeddings (mean-pooled Word2Vec vectors) to inspect separability in vector space.

### 3. Feature Engineering (`PySpark`)
- Tokenise with whitespace `Tokenizer` (punctuation and case preserved deliberately — they are stylistic signal).
- Build vocabulary at `min_count = 5` → **35,908 unique words**.
- Map each word to a 300-dimensional **GloVe 6B** vector. Coverage: **89.2%**. Out-of-vocabulary words receive small random initialisation and are fine-tuned during training.
- Convert every chunk to a fixed-length sequence of 200 integer token IDs.
- Pad/truncate to exactly 200 integers (rarely triggers — chunks are already 200 words).

### 4. Model Training (`TensorFlow / Keras`)

**Baseline BLSTM:**
- Embedding (300-d, GloVe-initialised, trainable)
- Bidirectional LSTM (128 units)
- Dropout (0.3)
- Dense (8 units, softmax)

Result: validation macro-F1 = **0.72**, but clear overfitting (training accuracy 99.9% vs validation 73%).

**Optimised BLSTM (Optuna):**
- Hyperparameter search over hidden units, embedding dropout, LSTM dropout, learning rate, and batch size.
- Best configuration: **256 hidden units, embedding dropout 0.5, LSTM dropout 0.3, learning rate 0.001, batch size 128**.
- Median pruner and a 6-hour wall-clock guard to keep the study within budget.
- Trained for 6 epochs (median best epoch across completed trials).

Result: validation macro-F1 = **0.74**, stable loss curve.

### 5. Evaluation

**Test set performance (final optimised model):**

| Metric | Value |
|---|---|
| Accuracy | 0.75 |
| Macro precision | 0.72 |
| Macro recall | 0.72 |
| Macro F1 | 0.70 |
| Weighted F1 | 0.74 |

**Per-author F1:**

| Author | F1 |
|---|---|
| Jonathan Swift | 0.96 |
| Jack London | 0.83 |
| Herman Melville | 0.79 |
| Fyodor Dostoyevsky | 0.75 |
| Arthur Conan Doyle | 0.66 |
| Bernard Shaw | 0.66 |
| Nathaniel Hawthorne | 0.57 |
| Leo Tolstoy | 0.38 |

**Key findings from the confusion matrix:**
- Tolstoy is heavily misclassified as Dostoyevsky — likely because both reach the corpus through English translation, and the model learned translation register rather than authorial style.
- Hawthorne scatters across Melville, Conan Doyle, and Dostoyevsky — reflecting genuine stylistic proximity between 19th-century authors in the same register.

---

## Project Structure

```
Authorship_TextAttribution/
├── author_classification.ipynb       # Full analysis notebook
├── dataset/                          # Raw Gutenberg texts, organised by author
│   ├── Arthur Conan Doyle/
│   ├── Bernard Shaw/
│   ├── ...
├── glove.6B.300d.txt                 # Pre-trained GloVe vectors
├── author_chunks_200/                # Parquet chunks dataset
├── trials/                           # Saved models per Optuna trial
├── authorship_final.keras            # Final trained model
└── README.md
```

---

## Setup

### Requirements

- **Python** 3.10+
- **Java** 21 (JDK, for PySpark)
- **Apache Spark / PySpark** 3.5+
- **Hadoop winutils** — required on Windows (see note below)
- **TensorFlow** 2.15+
- **Optuna** with `optuna-integration`

```bash
pip install pyspark tensorflow optuna optuna-integration \
            gensim scikit-learn matplotlib seaborn wordcloud pandas numpy
```

### Windows-specific setup

PySpark on Windows requires the Hadoop native utilities. Download `winutils.exe` and `hadoop.dll` matching your Spark's Hadoop version (e.g. from `cdarlint/winutils` on GitHub), then:

```python
import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
```

Set these **before** creating the `SparkSession`.

### Environment variables (performance)

To enable Intel oneDNN CPU acceleration in TensorFlow, set these **before** importing TensorFlow:

```python
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"   # match your vCPU count
```

A kernel restart is required for these to take effect.

---

## Running the Notebook

1. Place book files under `dataset/<Author Name>/<Title>.txt`.
2. Place `glove.6B.300d.txt` in the project root.
3. Open `author_classification.ipynb` and run cells top-to-bottom.
4. The notebook produces:
   - `author_chunks_200/` — Parquet dataset of 200-word chunks
   - `trials/` — one saved model per Optuna trial
   - `authorship_final.keras` — the final trained model
   - Training curves and confusion matrix plots

---

## Design Decisions Worth Noting

| Decision | Rationale |
|---|---|
| 200-word non-overlapping chunks | Long enough to capture style; short enough to yield many training examples |
| Split by **book**, not by chunk | Prevents data leakage from the same book appearing in train and test |
| Stratified split by author | Ensures minority authors appear in all partitions |
| No lemmatisation, no stop-word removal | Preserves stylistic surface features (function words, punctuation, capitalisation) that distinguish authors |
| GloVe embeddings, trainable | Provides a strong semantic prior; fine-tuning adapts vectors to the corpus |
| Macro-F1 as primary metric | Treats each author equally, regardless of chunk count |
| Early stopping on `val_macro_f1` | Aligns stopping with the reported objective, not a loss proxy |

---

## Known Limitations

- **Translation confusion.** Tolstoy and Dostoyevsky reach the corpus via English translation. The model partly learns translator style rather than original authorial style. Resolving this would require original-language texts.
- **Genuine stylistic overlap.** Hawthorne, Melville, and Conan Doyle share a 19th-century register that the model struggles to disentangle. This is not a data shortage problem — it reflects real stylistic proximity.
- **Corpus size.** Eight authors is small. Scaling to more authors would test whether the approach generalises.
- **No attention mechanism.** Mean-pooled and sequence-level representations do not explicitly separate style from topic. Attention could help.

---

## Deployment Recommendation

The model is accurate enough for a fan-facing bot provided it is deployed with three safeguards:

1. Frame the output as a **playful stylistic match**, not a factual authorship claim.
2. **Surface known confusions** (e.g. "this resembles Dostoyevsky or Tolstoy") rather than presenting a single confident answer that may be arbitrary.
3. Position the feature as a **novelty engagement tool**, not a forensic analysis service.

---

## Future Work

- Add original-language texts for the Russian authors to resolve translation confusion.
- Extend the author roster to test scalability.
- Replace mean-pooled representations with attention-based sequence encoders to separate style from topic.
- Experiment with character-level models, which are often competitive on authorship attribution and may capture punctuation rhythm more directly.

---

## Acknowledgements

- Book texts: [Project Gutenberg](https://www.gutenberg.org/)
- Pre-trained embeddings: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- Hadoop Windows binaries: `cdarlint/winutils` community repository

---

## License

Academic project. Text sources retain their original Project Gutenberg licensing.
