import Testing
@testable import CirceKit

/// Tests for the Swift port of OpenAI Whisper's English text normalizer.
///
/// The expected values are produced by the reference Python implementation
/// (`whisper_normalizer`) and cover casing/punctuation, contractions,
/// abbreviations, filler words, brackets, spelled-out numbers (cardinals,
/// ordinals, decimals, years, "one oh one", "double"), currency, percentages,
/// signs, British→American spelling, and diacritics.
struct EnglishTextNormalizerTests {
    let normalizer = EnglishTextNormalizer.bundled()

    struct Case: Sendable {
        let input: String
        let expected: String
        init(_ input: String, _ expected: String) {
            self.input = input
            self.expected = expected
        }
    }

    static let cases: [Case] = [
        // casing / punctuation / contractions
        Case("Unfortunately, studying traffic flow is difficult because drivers don't behave.",
             "unfortunately studying traffic flow is difficult because drivers do not behave"),
        Case("She won't go, and I can't either; let's leave.",
             "she will not go and i can not either let us leave"),
        Case("They've been waiting and I'd gone home.",
             "they have been waiting and i had gone home"),
        Case("It's John's book, isn't it?",
             "it is john is book is not it"),
        // abbreviations / titles
        Case("Dr. Smith and Mr. Jones met with Prof. Adams.",
             "doctor smith and mister jones met with professor adams"),
        // filler words + brackets/parentheses
        Case("um so I was like uh going to the store",
             "so i was like going to the store"),
        Case("[music playing] the concert was great (loud applause)",
             "the concert was great"),
        // numbers: cardinal / ordinal / decimals / years
        Case("He scored one hundred and twenty three points.",
             "he scored 123 points"),
        Case("She finished third, just behind the second place runner.",
             "she finished 3rd just behind the 2nd place runner"),
        Case("The recipe needs two and a half cups of flour.",
             "the recipe needs 2.5 cups of flour"),
        Case("Room one oh one is down the hall.",
             "room 101 is down the hall"),
        Case("It happened in nineteen sixty four.",
             "it happened in 1964"),
        Case("The value is three point five.",
             "the value is 3.5"),
        // currency / percent
        Case("I paid twenty three dollars and fifty cents.",
             "i paid $23.50"),
        Case("That is about 3.5 million dollars in total.",
             "that is about $3500000 in total"),
        Case("Approximately ninety percent agreed.",
             "approximately 90% agreed"),
        Case("It costs fifteen percent more.",
             "it costs 15% more"),
        // signs / nominal digit sequences
        Case("The temperature dropped to minus five degrees.",
             "the temperature dropped to -5 degrees"),
        Case("Call me at five five five one two three four.",
             "call me at 5551234"),
        // double / triple
        Case("The double oh seven film is famous.",
             "the 007 film is famous"),
        // British -> American spelling
        Case("Colour, flavour and honour are spelled differently.",
             "color flavor and honor are spelled differently"),
        Case("They recognised the organisation's centre.",
             "they recognized the organization is center"),
        // diacritics
        Case("naïve café résumé",
             "naive cafe resume"),
        // digit suffixes preserved
        Case("the 1960s and the 274th day",
             "the 1960s and the 274th day"),
        // already-normalized FLEURS-style line (numbers still standardized)
        Case("twentieth century research has shown that there are two pools of memory",
             "20th century research has shown that there are 2 pools of memory"),
    ]

    @Test(arguments: cases)
    func matchesWhisperReference(_ testCase: Case) {
        #expect(normalizer.normalize(testCase.input) == testCase.expected)
    }

    @Test func emptyStringNormalizesToEmpty() {
        #expect(normalizer.normalize("") == "")
    }

    @Test func whitespaceOnlyNormalizesToEmpty() {
        #expect(normalizer.normalize("   \t  \n ") == "")
    }

    @Test func spellingMapIsLoaded() {
        // A British→American mapping means the bundled english.json was found.
        #expect(normalizer.normalize("colour") == "color")
    }
}
