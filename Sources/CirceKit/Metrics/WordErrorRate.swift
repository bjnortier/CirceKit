import Foundation

/// Word Error Rate: the word-level edit distance between a reference and a
/// hypothesis transcription, divided by the reference word count.
///
/// Comparing engines needs this: whisper.cpp writes `"123"` where Apple writes
/// `"one hundred twenty three"`, so raw string equality is meaningless across
/// backends. Pass ``EnglishTextNormalizer`` as the tokenizer for English.
public nonisolated enum WordErrorRate {
    /// Word-error measurement for a single sample. Keeps the raw counts so that
    /// results can be aggregated at the corpus level (total errors / total words)
    /// rather than by averaging per-sample rates.
    public struct Measurement: Sendable, Equatable {
        /// Word-level edit distance (substitutions + deletions + insertions).
        public let errors: Int
        /// Number of words in the reference.
        public let referenceWords: Int

        public init(errors: Int, referenceWords: Int) {
            self.errors = errors
            self.referenceWords = referenceWords
        }

        /// This sample's own WER. An empty reference scores 0 if the hypothesis
        /// is also empty, otherwise 1.
        public var rate: Double {
            guard referenceWords > 0 else { return errors == 0 ? 0 : 1 }
            return Double(errors) / Double(referenceWords)
        }
    }

    /// Measures errors and reference length using a caller-supplied tokenizer,
    /// e.g. the OpenAI English normalizer for English locales.
    public static func measure(
        reference: String,
        hypothesis: String,
        tokenizer: (String) -> [String]
    ) -> Measurement {
        let referenceWords = tokenizer(reference)
        let hypothesisWords = tokenizer(hypothesis)
        return Measurement(
            errors: editDistance(referenceWords, hypothesisWords),
            referenceWords: referenceWords.count
        )
    }

    /// Computes WER using the default normalization (lowercase + strip punctuation).
    public static func compute(reference: String, hypothesis: String) -> Double {
        compute(reference: reference, hypothesis: hypothesis, tokenizer: tokenize)
    }

    /// Computes WER using a caller-supplied tokenizer.
    public static func compute(
        reference: String,
        hypothesis: String,
        tokenizer: (String) -> [String]
    ) -> Double {
        measure(reference: reference, hypothesis: hypothesis, tokenizer: tokenizer).rate
    }

    /// Aggregates per-sample measurements into a corpus WER.
    ///
    /// This is Σerrors / Σreference-words, deliberately not the mean of the
    /// per-sample rates: short samples would otherwise dominate the result.
    public static func corpusRate(_ measurements: some Sequence<Measurement>) -> Double {
        var errors = 0
        var words = 0
        for measurement in measurements {
            errors += measurement.errors
            words += measurement.referenceWords
        }
        guard words > 0 else { return errors == 0 ? 0 : 1 }
        return Double(errors) / Double(words)
    }

    /// Lowercases, strips anything that isn't alphanumeric, and splits into words.
    public static func tokenize(_ text: String) -> [String] {
        var scalars = String.UnicodeScalarView()
        for scalar in text.lowercased().unicodeScalars {
            scalars.append(CharacterSet.alphanumerics.contains(scalar) ? scalar : " ")
        }
        return String(scalars).split(whereSeparator: { $0 == " " }).map(String.init)
    }

    /// Levenshtein distance over word arrays (substitution/insertion/deletion).
    public static func editDistance(_ a: [String], _ b: [String]) -> Int {
        let n = a.count, m = b.count
        if n == 0 { return m }
        if m == 0 { return n }

        var previous = Array(0...m)
        var current = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            current[0] = i
            for j in 1...m {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                current[j] = Swift.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            }
            swap(&previous, &current)
        }
        return previous[m]
    }
}
