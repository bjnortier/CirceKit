import Testing
@testable import CirceKit

struct WordErrorRateTests {
    private func isClose(_ a: Double, _ b: Double) -> Bool { abs(a - b) < 1e-9 }

    @Test func identicalIsZero() {
        #expect(WordErrorRate.compute(reference: "the cat sat", hypothesis: "the cat sat") == 0)
    }

    @Test func casingAndPunctuationIgnored() {
        #expect(WordErrorRate.compute(reference: "the cat sat", hypothesis: "The, CAT sat!") == 0)
    }

    @Test func oneSubstitutionOfThree() {
        #expect(isClose(WordErrorRate.compute(reference: "the cat sat", hypothesis: "the dog sat"), 1.0 / 3.0))
    }

    @Test func oneDeletionOfThree() {
        #expect(isClose(WordErrorRate.compute(reference: "the cat sat", hypothesis: "the sat"), 1.0 / 3.0))
    }

    @Test func oneInsertionOfThree() {
        #expect(isClose(WordErrorRate.compute(reference: "the cat sat", hypothesis: "the big cat sat"), 1.0 / 3.0))
    }

    @Test func emptyHypothesisIsFullError() {
        #expect(WordErrorRate.compute(reference: "the cat sat", hypothesis: "") == 1)
    }

    /// With the English normalizer, differently-formatted numbers score as a match.
    @Test func englishNormalizerMakesNumberFormattingMatch() {
        let normalizer = EnglishTextNormalizer.bundled()
        let tokenizer: (String) -> [String] = {
            normalizer.normalize($0).split(whereSeparator: { $0.isWhitespace }).map(String.init)
        }
        let wer = WordErrorRate.compute(
            reference: "he scored one hundred and twenty three points",
            hypothesis: "He scored 123 points.",
            tokenizer: tokenizer
        )
        #expect(wer == 0)
    }
}
