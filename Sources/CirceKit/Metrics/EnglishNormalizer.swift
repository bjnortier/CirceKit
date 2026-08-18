import Foundation

/// Swift port of OpenAI Whisper's English text normalizer
/// (`whisper_normalizer/english.py`), used to make WER scoring robust to
/// differences in casing, punctuation, number formatting, and British vs.
/// American spelling.

// MARK: - Rational number (subset of Python's Fraction used by the normalizer)

private nonisolated struct Fraction: Sendable {
    let numerator: Int
    let denominator: Int

    /// Parses a string matching `\d+(\.\d+)?`.
    init?(_ s: String) {
        if let dot = s.firstIndex(of: ".") {
            let intPart = String(s[..<dot])
            let fracPart = String(s[s.index(after: dot)...])
            guard let ip = Int(intPart), let fp = Int(fracPart) else { return nil }
            var denom = 1
            for _ in 0..<fracPart.count { denom *= 10 }
            self.init(ip * denom + fp, denom)
        } else {
            guard let n = Int(s) else { return nil }
            self.init(n, 1)
        }
    }

    init(_ n: Int, _ d: Int) {
        let g = max(1, Fraction.gcd(abs(n), abs(d)))
        numerator = n / g
        denominator = d / g
    }

    func times(_ m: Int) -> Fraction { Fraction(numerator * m, denominator) }

    private static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
}

// MARK: - Regex helpers

private nonisolated func regexReplace(_ s: String, _ pattern: String, _ template: String) -> String {
    guard let re = try? NSRegularExpression(pattern: pattern) else { return s }
    let range = NSRange(s.startIndex..., in: s)
    return re.stringByReplacingMatches(in: s, options: [], range: range, withTemplate: template)
}

private nonisolated func isDecimalNumber(_ s: String) -> Bool {
    guard !s.isEmpty else { return false }
    var seenDot = false
    var digitsBeforeDot = false
    var digitsAfterDot = false
    for ch in s {
        if ch == "." {
            if seenDot || !digitsBeforeDot { return false }
            seenDot = true
        } else if ch.isNumber && ch.isASCII {
            if seenDot { digitsAfterDot = true } else { digitsBeforeDot = true }
        } else {
            return false
        }
    }
    return digitsBeforeDot && (!seenDot || digitsAfterDot)
}

// MARK: - Symbol / diacritic removal (from basic.py)

/// Replaces markers/symbols/punctuation with a space and drops diacritics,
/// keeping any scalars in `keep`.
private nonisolated func removeSymbolsAndDiacritics(_ s: String, keep: Set<Unicode.Scalar>) -> String {
    // non-ASCII letters not separated by NFKD normalization
    let additionalDiacritics: [Unicode.Scalar: String] = [
        "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
        "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
    ]
    var result = ""
    for scalar in s.decomposedStringWithCompatibilityMapping.unicodeScalars {
        if keep.contains(scalar) {
            result.unicodeScalars.append(scalar)
        } else if let replacement = additionalDiacritics[scalar] {
            result += replacement
        } else {
            switch scalar.properties.generalCategory {
            case .nonspacingMark:
                break // drop
            case .spacingMark, .enclosingMark,
                 .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
                 .connectorPunctuation, .dashPunctuation, .openPunctuation,
                 .closePunctuation, .initialPunctuation, .finalPunctuation, .otherPunctuation:
                result += " "
            default:
                result.unicodeScalars.append(scalar)
            }
        }
    }
    return result
}

// MARK: - Number normalizer

private nonisolated final class EnglishNumberNormalizer: Sendable {
    private enum Val: Equatable {
        case none
        case int(Int)
        case str(String)
    }

    private let zeros: Set<String> = ["o", "oh", "zero"]
    private let ones: [String: Int]
    private let onesSuffixed: [String: (Int, String)]
    private let tens: [String: Int]
    private let tensSuffixed: [String: (Int, String)]
    private let multipliers: [String: Int]
    private let multipliersSuffixed: [String: (Int, String)]
    private let decimals: Set<String>
    private let precedingPrefixers: [String: String]
    private let followingPrefixers: [String: String]
    private let prefixes: Set<String>
    private let suffixers: [String: SuffixValue]
    private let specials: Set<String> = ["and", "double", "triple", "point"]
    private let words: Set<String>

    private enum SuffixValue { case string(String); case map([String: String]) }

    init() {
        let onesNames = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                         "seventeen", "eighteen", "nineteen"]
        var onesDict: [String: Int] = [:]
        for (i, name) in onesNames.enumerated() { onesDict[name] = i + 1 }
        ones = onesDict

        var onesPlural: [String: (Int, String)] = [:]
        for (name, value) in onesDict {
            let key = name == "six" ? "sixes" : name + "s"
            onesPlural[key] = (value, "s")
        }
        var onesOrdinal: [String: (Int, String)] = [
            "zeroth": (0, "th"), "first": (1, "st"), "second": (2, "nd"),
            "third": (3, "rd"), "fifth": (5, "th"), "twelfth": (12, "th"),
        ]
        for (name, value) in onesDict where value > 3 && value != 5 && value != 12 {
            let key = name + (name.hasSuffix("t") ? "h" : "th")
            onesOrdinal[key] = (value, "th")
        }
        onesSuffixed = onesPlural.merging(onesOrdinal) { _, new in new }

        let tensDict: [String: Int] = [
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        ]
        tens = tensDict
        var tensSuffixedDict: [String: (Int, String)] = [:]
        for (name, value) in tensDict {
            tensSuffixedDict[name.replacingOccurrences(of: "y", with: "ies")] = (value, "s")
            tensSuffixedDict[name.replacingOccurrences(of: "y", with: "ieth")] = (value, "th")
        }
        tensSuffixed = tensSuffixedDict

        let multipliersDict: [String: Int] = [
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
        ]
        multipliers = multipliersDict
        var multipliersSuffixedDict: [String: (Int, String)] = [:]
        for (name, value) in multipliersDict {
            multipliersSuffixedDict[name + "s"] = (value, "s")
            multipliersSuffixedDict[name + "th"] = (value, "th")
        }
        multipliersSuffixed = multipliersSuffixedDict

        decimals = Set(onesDict.keys).union(tensDict.keys).union(zeros)

        precedingPrefixers = ["minus": "-", "negative": "-", "plus": "+", "positive": "+"]
        followingPrefixers = [
            "pound": "£", "pounds": "£", "euro": "€", "euros": "€",
            "dollar": "$", "dollars": "$", "cent": "¢", "cents": "¢",
        ]
        prefixes = Set(precedingPrefixers.values).union(followingPrefixers.values)
        suffixers = ["per": .map(["cent": "%"]), "percent": .string("%")]

        var allWords: Set<String> = []
        allWords.formUnion(zeros)
        allWords.formUnion(onesDict.keys)
        allWords.formUnion(onesSuffixed.keys)
        allWords.formUnion(tensDict.keys)
        allWords.formUnion(tensSuffixed.keys)
        allWords.formUnion(multipliersDict.keys)
        allWords.formUnion(multipliersSuffixed.keys)
        allWords.formUnion(precedingPrefixers.keys)
        allWords.formUnion(followingPrefixers.keys)
        allWords.formUnion(suffixers.keys)
        allWords.formUnion(specials)
        words = allWords
    }

    func callAsFunction(_ s: String) -> String {
        let pre = preprocess(s)
        let processed = processWords(pre.split(whereSeparator: { $0.isWhitespace }).map(String.init))
        return postprocess(processed.joined(separator: " "))
    }

    // MARK: Value helpers

    private func isNone(_ v: Val) -> Bool { v == .none }
    private func isStr(_ v: Val) -> Bool { if case .str = v { return true }; return false }
    private func intVal(_ v: Val) -> Int { if case .int(let n) = v { return n }; return 0 }
    private func valStr(_ v: Val) -> String {
        switch v {
        case .none: return ""
        case .int(let n): return String(n)
        case .str(let s): return s
        }
    }
    /// Mirrors Python `str(value or "")`.
    private func truthy(_ v: Val) -> String {
        switch v {
        case .none: return ""
        case .int(let n): return n == 0 ? "" : String(n)
        case .str(let s): return s
        }
    }
    private func contains(_ set: Set<String>, _ word: String?) -> Bool {
        guard let word else { return false }
        return set.contains(word)
    }
    private func inKeys(_ dict: [String: Int], _ word: String?) -> Bool {
        guard let word else { return false }
        return dict[word] != nil
    }

    // MARK: Core

    private func processWords(_ words: [String]) -> [String] {
        var results: [String] = []
        var prefix: String? = nil
        var value: Val = .none
        var skip = false

        func output(_ result: String) {
            var r = result
            if let p = prefix { r = p + r }
            value = .none
            prefix = nil
            results.append(r)
        }

        guard !words.isEmpty else { return results }

        var i = 0
        while i < words.count {
            defer { i += 1 }
            if skip { skip = false; continue }

            let prev: String? = i > 0 ? words[i - 1] : nil
            let current = words[i]
            let next: String? = i < words.count - 1 ? words[i + 1] : nil
            let nextIsNumeric = next.map(isDecimalNumber) ?? false
            let hasPrefix = current.first.map { prefixes.contains(String($0)) } ?? false
            let currentWithoutPrefix = hasPrefix ? String(current.dropFirst()) : current

            if isDecimalNumber(currentWithoutPrefix) {
                guard let f = Fraction(currentWithoutPrefix) else { continue }
                if !isNone(value) {
                    if case .str(let sv) = value, sv.hasSuffix(".") {
                        value = .str(sv + current)
                        continue
                    } else {
                        output(valStr(value))
                    }
                }
                if hasPrefix { prefix = String(current.first!) }
                if f.denominator == 1 { value = .int(f.numerator) } else { value = .str(currentWithoutPrefix) }
            } else if !self.words.contains(current) {
                if !isNone(value) { output(valStr(value)) }
                output(current)
            } else if zeros.contains(current) {
                value = .str(truthy(value) + "0")
            } else if let onesVal = ones[current] {
                if isNone(value) {
                    value = .int(onesVal)
                } else if isStr(value) || inKeys(ones, prev) {
                    if inKeys(tens, prev) && onesVal < 10 {
                        value = .str(String(valStr(value).dropLast()) + String(onesVal))
                    } else {
                        value = .str(valStr(value) + String(onesVal))
                    }
                } else if onesVal < 10 {
                    if intVal(value) % 10 == 0 { value = .int(intVal(value) + onesVal) }
                    else { value = .str(String(intVal(value)) + String(onesVal)) }
                } else {
                    if intVal(value) % 100 == 0 { value = .int(intVal(value) + onesVal) }
                    else { value = .str(String(intVal(value)) + String(onesVal)) }
                }
            } else if let (onesVal, suffix) = onesSuffixed[current] {
                if isNone(value) {
                    output(String(onesVal) + suffix)
                } else if isStr(value) || inKeys(ones, prev) {
                    if inKeys(tens, prev) && onesVal < 10 {
                        output(String(valStr(value).dropLast()) + String(onesVal) + suffix)
                    } else {
                        output(valStr(value) + String(onesVal) + suffix)
                    }
                } else if onesVal < 10 {
                    if intVal(value) % 10 == 0 { output(String(intVal(value) + onesVal) + suffix) }
                    else { output(String(intVal(value)) + String(onesVal) + suffix) }
                } else {
                    if intVal(value) % 100 == 0 { output(String(intVal(value) + onesVal) + suffix) }
                    else { output(String(intVal(value)) + String(onesVal) + suffix) }
                }
                value = .none
            } else if let tensVal = tens[current] {
                if isNone(value) {
                    value = .int(tensVal)
                } else if isStr(value) {
                    value = .str(valStr(value) + String(tensVal))
                } else {
                    if intVal(value) % 100 == 0 { value = .int(intVal(value) + tensVal) }
                    else { value = .str(String(intVal(value)) + String(tensVal)) }
                }
            } else if let (tensVal, suffix) = tensSuffixed[current] {
                if isNone(value) {
                    output(String(tensVal) + suffix)
                } else if isStr(value) {
                    output(valStr(value) + String(tensVal) + suffix)
                } else {
                    if intVal(value) % 100 == 0 { output(String(intVal(value) + tensVal) + suffix) }
                    else { output(String(intVal(value)) + String(tensVal) + suffix) }
                }
            } else if let multiplier = multipliers[current] {
                if isNone(value) {
                    value = .int(multiplier)
                } else if isStr(value) || intVal(value) == 0 {
                    let f = Fraction(truthyOrZero(value))
                    if let f, f.times(multiplier).denominator == 1 {
                        value = .int(f.times(multiplier).numerator)
                    } else {
                        output(valStr(value))
                        value = .int(multiplier)
                    }
                } else {
                    let before = intVal(value) / 1000 * 1000
                    let residual = intVal(value) % 1000
                    value = .int(before + residual * multiplier)
                }
            } else if let (multiplier, suffix) = multipliersSuffixed[current] {
                if isNone(value) {
                    output(String(multiplier) + suffix)
                } else if isStr(value) {
                    let f = Fraction(valStr(value))
                    if let f, f.times(multiplier).denominator == 1 {
                        output(String(f.times(multiplier).numerator) + suffix)
                    } else {
                        output(valStr(value))
                        output(String(multiplier) + suffix)
                    }
                } else {
                    let before = intVal(value) / 1000 * 1000
                    let residual = intVal(value) % 1000
                    value = .int(before + residual * multiplier)
                    output(valStr(value) + suffix)
                }
                value = .none
            } else if let symbol = precedingPrefixers[current] {
                if !isNone(value) { output(valStr(value)) }
                if contains(self.words, next) || nextIsNumeric {
                    prefix = symbol
                } else {
                    output(current)
                }
            } else if let symbol = followingPrefixers[current] {
                if !isNone(value) {
                    prefix = symbol
                    output(valStr(value))
                } else {
                    output(current)
                }
            } else if let suffixer = suffixers[current] {
                if !isNone(value) {
                    switch suffixer {
                    case .map(let m):
                        if let n = next, let mapped = m[n] {
                            output(valStr(value) + mapped)
                            skip = true
                        } else {
                            output(valStr(value))
                            output(current)
                        }
                    case .string(let suffix):
                        output(valStr(value) + suffix)
                    }
                } else {
                    output(current)
                }
            } else if specials.contains(current) {
                if !contains(self.words, next) && !nextIsNumeric {
                    if !isNone(value) { output(valStr(value)) }
                    output(current)
                } else if current == "and" {
                    if !inKeys(multipliers, prev) {
                        if !isNone(value) { output(valStr(value)) }
                        output(current)
                    }
                } else if current == "double" || current == "triple" {
                    if let n = next, ones[n] != nil || zeros.contains(n) {
                        let repeats = current == "double" ? 2 : 3
                        let onesVal = ones[n] ?? 0
                        value = .str(truthy(value) + String(repeating: String(onesVal), count: repeats))
                        skip = true
                    } else {
                        if !isNone(value) { output(valStr(value)) }
                        output(current)
                    }
                } else if current == "point" {
                    if let n = next, decimals.contains(n) || nextIsNumeric {
                        value = .str(truthy(value) + ".")
                    }
                }
            }
        }

        if !isNone(value) { output(valStr(value)) }
        return results
    }

    /// Mirrors `to_fraction(value)` where value may be a string or int 0.
    private func truthyOrZero(_ v: Val) -> String {
        switch v {
        case .none: return "0"
        case .int(let n): return String(n)
        case .str(let s): return s
        }
    }

    // MARK: Pre/post-processing

    private func preprocess(_ input: String) -> String {
        var results: [String] = []
        let segments = splitOnAndAHalf(input)
        for (i, segment) in segments.enumerated() {
            if segment.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            if i == segments.count - 1 {
                results.append(segment)
            } else {
                results.append(segment)
                let lastWord = segment.split(whereSeparator: { $0.isWhitespace }).last.map(String.init) ?? ""
                if decimals.contains(lastWord) || multipliers[lastWord] != nil {
                    results.append("point five")
                } else {
                    results.append("and a half")
                }
            }
        }
        var s = results.joined(separator: " ")
        s = regexReplace(s, "([a-z])([0-9])", "$1 $2")
        s = regexReplace(s, "([0-9])([a-z])", "$1 $2")
        s = regexReplace(s, "([0-9])\\s+(st|nd|rd|th|s)\\b", "$1$2")
        return s
    }

    private func splitOnAndAHalf(_ s: String) -> [String] {
        guard let re = try? NSRegularExpression(pattern: "\\band\\s+a\\s+half\\b") else { return [s] }
        let full = NSRange(s.startIndex..., in: s)
        let matches = re.matches(in: s, options: [], range: full)
        guard !matches.isEmpty else { return [s] }
        var segments: [String] = []
        var last = s.startIndex
        for match in matches {
            guard let r = Range(match.range, in: s) else { continue }
            segments.append(String(s[last..<r.lowerBound]))
            last = r.upperBound
        }
        segments.append(String(s[last...]))
        return segments
    }

    private func postprocess(_ input: String) -> String {
        var s = input
        // "$2 and ¢7" -> "$2.07"
        s = replaceMatches(s, "([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\\b") { groups in
            guard let cents = Int(groups[3]) else { return nil }
            return groups[1] + groups[2] + "." + String(format: "%02d", cents)
        }
        // "€0.07" -> "¢7"
        s = replaceMatches(s, "[€£$]0.([0-9]{1,2})\\b") { groups in
            guard let n = Int(groups[1]) else { return nil }
            return "¢" + String(n)
        }
        // write "one(s)" instead of "1(s)"
        s = regexReplace(s, "\\b1(s?)\\b", "one$1")
        return s
    }

    /// Applies a function-based regex replacement. The closure receives capture
    /// groups (index 0 = whole match); returning nil keeps the original match.
    private func replaceMatches(
        _ s: String,
        _ pattern: String,
        _ transform: ([String]) -> String?
    ) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return s }
        let full = NSRange(s.startIndex..., in: s)
        var result = ""
        var last = s.startIndex
        for match in re.matches(in: s, options: [], range: full) {
            guard let matchRange = Range(match.range, in: s) else { continue }
            var groups: [String] = []
            for g in 0..<match.numberOfRanges {
                if let gr = Range(match.range(at: g), in: s) {
                    groups.append(String(s[gr]))
                } else {
                    groups.append("")
                }
            }
            result += s[last..<matchRange.lowerBound]
            result += transform(groups) ?? String(s[matchRange])
            last = matchRange.upperBound
        }
        result += s[last...]
        return result
    }
}

// MARK: - Spelling normalizer

private nonisolated struct EnglishSpellingNormalizer: Sendable {
    let mapping: [String: String]

    func callAsFunction(_ s: String) -> String {
        s.split(whereSeparator: { $0.isWhitespace })
            .map { mapping[String($0)] ?? String($0) }
            .joined(separator: " ")
    }
}

// MARK: - Full English text normalizer

/// A Swift port of OpenAI Whisper's `whisper_normalizer/english.py`.
///
/// Folds away the differences that make cross-engine transcript comparison
/// unfair — digits vs words, contractions, titles, fillers, British vs American
/// spelling — so that ``WordErrorRate`` measures recognition errors rather than
/// formatting choices.
public nonisolated struct EnglishTextNormalizer: Sendable {
    private let numbers = EnglishNumberNormalizer()
    private let spellings: EnglishSpellingNormalizer

    private static let ignorePattern = "\\b(hmm|mm|mhm|mmm|uh|um)\\b"
    private static let keep: Set<Unicode.Scalar> = Set(".%$¢€£".unicodeScalars)

    /// Ordered contraction / abbreviation replacements (order matters).
    private static let replacers: [(String, String)] = [
        ("\\bwon't\\b", "will not"), ("\\bcan't\\b", "can not"), ("\\blet's\\b", "let us"),
        ("\\bain't\\b", "aint"), ("\\by'all\\b", "you all"), ("\\bwanna\\b", "want to"),
        ("\\bkinda\\b", "kind of"), ("\\bsorta\\b", "sort of"), ("\\bdunno\\b", "do not know"),
        ("\\bgotta\\b", "got to"), ("\\bgonna\\b", "going to"), ("\\bi'ma\\b", "i am going to"),
        ("\\bimma\\b", "i am going to"), ("\\bwoulda\\b", "would have"), ("\\bcoulda\\b", "could have"),
        ("\\bshoulda\\b", "should have"), ("\\bcause\\b", "because"), ("\\bma'am\\b", "madam"),
        ("\\bmr\\b", "mister "), ("\\bmrs\\b", "missus "), ("\\bst\\b", "saint "),
        ("\\bdr\\b", "doctor "), ("\\bprof\\b", "professor "), ("\\bcapt\\b", "captain "),
        ("\\bgov\\b", "governor "), ("\\bald\\b", "alderman "), ("\\bgen\\b", "general "),
        ("\\bsen\\b", "senator "), ("\\brep\\b", "representative "), ("\\bpres\\b", "president "),
        ("\\brev\\b", "reverend "), ("\\bhon\\b", "honorable "), ("\\basst\\b", "assistant "),
        ("\\bassoc\\b", "associate "), ("\\blt\\b", "lieutenant "), ("\\bcol\\b", "colonel "),
        ("\\bjr\\b", "junior "), ("\\bsr\\b", "senior "), ("\\besq\\b", "esquire "),
        ("'d been\\b", " had been"), ("'s been\\b", " has been"), ("'d gone\\b", " had gone"),
        ("'s gone\\b", " has gone"), ("'d done\\b", " had done"), ("'s got\\b", " has got"),
        ("n't\\b", " not"), ("'re\\b", " are"), ("'s\\b", " is"), ("'d\\b", " would"),
        ("'ll\\b", " will"), ("'t\\b", " not"), ("'ve\\b", " have"), ("'m\\b", " am"),
    ]

    public init(spellingMapping: [String: String]) {
        spellings = EnglishSpellingNormalizer(mapping: spellingMapping)
    }

    /// The shared normalizer, using the bundled spelling map.
    public static let shared = EnglishTextNormalizer.bundled()

    /// Loads the bundled British→American spelling map. Falls back to an empty
    /// map if the resource is missing.
    public static func bundled() -> EnglishTextNormalizer {
        var mapping: [String: String] = [:]
        if let url = Bundle.module.url(forResource: "english", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
            mapping = decoded
        }
        return EnglishTextNormalizer(spellingMapping: mapping)
    }

    /// Normalizes then splits into words — the tokenizer form ``WordErrorRate``
    /// expects for English.
    public func tokenize(_ input: String) -> [String] {
        normalize(input).split(whereSeparator: { $0 == " " }).map(String.init)
    }

    public func normalize(_ input: String) -> String {
        var s = input.lowercased()

        s = regexReplace(s, "[<\\[][^>\\]]*[>\\]]", "")  // remove between brackets
        s = regexReplace(s, "\\(([^)]+?)\\)", "")        // remove between parentheses
        s = regexReplace(s, Self.ignorePattern, "")
        s = regexReplace(s, "\\s+'", "'")

        for (pattern, replacement) in Self.replacers {
            s = regexReplace(s, pattern, replacement)
        }

        s = regexReplace(s, "(\\d),(\\d)", "$1$2")       // remove commas between digits
        s = regexReplace(s, "\\.([^0-9]|$)", " $1")      // periods not followed by numbers
        s = removeSymbolsAndDiacritics(s, keep: Self.keep)

        s = numbers(s)
        s = spellings(s)

        s = regexReplace(s, "[.$¢€£]([^0-9])", " $1")
        s = regexReplace(s, "([^0-9])%", "$1 ")
        s = regexReplace(s, "\\s+", " ")

        return s
    }
}
