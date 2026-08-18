import CoreAI
import Foundation

/// A Core AI speech model bundle.
///
/// Unlike ``WhisperModel`` these are not downloadable — they are multi-gigabyte
/// export directories produced by the `coreai-models` repository. The named
/// cases resolve against a local exports directory; use ``bundle(_:)`` to point
/// at one directly.
public enum CoreAIModel: Sendable, Equatable, Hashable {
    /// Parakeet TDT 0.6B v3, float32, static 30 s encoder.
    case parakeetTDT06BV3

    /// Whisper large-v3-turbo with KV cache, float16.
    case whisperLargeV3Turbo

    /// An explicit bundle directory.
    case bundle(URL)

    /// Environment variable naming the directory holding exported bundles.
    public static let exportsDirectoryEnvironmentKey = "CIRCEKIT_COREAI_EXPORTS"

    /// Default location of the `coreai-models` exports directory.
    public static let defaultExportsDirectory = URL(
        filePath: "/Users/bjnortier/development/apple/coreai-models.bjnortier/exports",
        directoryHint: .isDirectory
    )

    /// The exports directory in effect, honouring the environment override.
    public static var exportsDirectory: URL {
        if let path = ProcessInfo.processInfo.environment[exportsDirectoryEnvironmentKey],
           !path.isEmpty {
            return URL(filePath: path, directoryHint: .isDirectory)
        }
        return defaultExportsDirectory
    }

    /// The bundle directory name within the exports directory, for named cases.
    public var bundleName: String? {
        switch self {
        case .parakeetTDT06BV3: return "parakeet-tdt-0.6b-v3_float32_static"
        case .whisperLargeV3Turbo: return "whisper-large-v3-turbo-kv_float16"
        case .bundle: return nil
        }
    }

    /// The bundle directory on disk, or `nil` if it isn't there.
    public var resolvedURL: URL? {
        let candidate: URL
        switch self {
        case .bundle(let url): candidate = url
        case .parakeetTDT06BV3, .whisperLargeV3Turbo:
            guard let bundleName else { return nil }
            candidate = Self.exportsDirectory.appending(path: bundleName, directoryHint: .isDirectory)
        }
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: candidate.path(percentEncoded: false),
            isDirectory: &isDirectory
        )
        return exists && isDirectory.boolValue ? candidate : nil
    }

    /// The bundle directory on disk, throwing if absent.
    public func requireURL() throws -> URL {
        guard let resolvedURL else {
            let name = bundleName ?? "bundle"
            throw CirceError.modelUnavailable(
                """
                Core AI bundle '\(name)' not found under \(Self.exportsDirectory.path(percentEncoded: false)). \
                Set \(Self.exportsDirectoryEnvironmentKey) or use CoreAIModel.bundle(_:).
                """
            )
        }
        return resolvedURL
    }
}

// MARK: - Compilation cache

/// How much of a Core AI bundle already has compiled artifacts on disk.
///
/// Reported as a count rather than a flag because the probe cannot see every
/// component: Core AI keys cached artifacts by the specialization options they
/// were built with, and `CoreAISpeech` loads a bundle's parts with *different*
/// compute units (its default is `encoder: .automatic, decoder: .cpu`). Probing
/// with one set of options therefore finds the encoder but misses the decoder
/// and joint, even on a thoroughly warm machine.
///
/// The encoder is the expensive component, so `cachedAssets == 0` is the signal
/// that matters: it means a genuinely cold run whose model load will be slow.
public struct CoreAICacheState: Sendable, Equatable {
    /// Components with a cached artifact the probe could see.
    public let cachedAssets: Int
    /// Components in the bundle.
    public let totalAssets: Int

    /// No component is cached — the next load compiles from scratch.
    public var isCold: Bool { totalAssets > 0 && cachedAssets == 0 }

    /// A short description for logs and result tables.
    public var summary: String {
        guard totalAssets > 0 else { return "unknown" }
        return isCold ? "cold" : "warm (\(cachedAssets)/\(totalAssets))"
    }
}

extension CoreAIModel {
    /// How much of this bundle is already compiled.
    ///
    /// Only inspects the cache; never triggers compilation. Compilation itself
    /// happens during model load — inside `prepare()`, outside any transcription
    /// timing — but a benchmark still wants to say which kind of run it measured.
    public var cacheState: CoreAICacheState {
        guard let url = resolvedURL else { return CoreAICacheState(cachedAssets: 0, totalAssets: 0) }
        let assets = Self.assetURLs(in: url)
        let cached = assets.count { asset in
            ((try? AIModelCache.default.model(for: asset, options: .default)) ?? nil) != nil
        }
        return CoreAICacheState(cachedAssets: cached, totalAssets: assets.count)
    }

    /// The Core AI asset components inside a bundle directory.
    ///
    /// Component filenames vary by model family — a Parakeet bundle prefixes each
    /// asset with the variant name — so scan by extension rather than assuming names.
    private static func assetURLs(in url: URL) -> [URL] {
        let assetExtensions: Set<String> = ["aimodel", "aimodelc"]
        if assetExtensions.contains(url.pathExtension) { return [url] }
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil
        )) ?? []
        return entries.filter { assetExtensions.contains($0.pathExtension) }.sorted { $0.path < $1.path }
    }
}

// MARK: - Forced language

extension CoreAIModel {
    /// The language this export pins, or `nil` if it transcribes any language.
    ///
    /// A Whisper export bakes its decoder prefix into `generation_config.json` —
    /// `<|startoftranscript|><|en|><|transcribe|>` — and `CoreAISpeech` offers no
    /// way to change it at run time. Fed French audio, an `<|en|>` export does not
    /// fail; it transcribes *as if* the audio were English, which reads as a
    /// plausible English sentence and scores like a broken model (96% WER on a
    /// French clip). Reporting the pin lets a caller skip the pairing instead of
    /// publishing a meaningless number.
    ///
    /// Parakeet TDT is inherently multilingual and pins nothing, so this is `nil`.
    public var forcedLanguage: Locale? {
        guard let url = resolvedURL else { return nil }

        // The prefix names a token id; the bundle's own token table names the
        // token. Reading both avoids hardcoding Whisper's language-token order,
        // which differs between model generations.
        guard
            let configData = try? Data(contentsOf: url.appending(path: "generation_config.json")),
            let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
            let forced = config["forced_decoder_ids"] as? [[Int]],
            let tokensData = try? Data(contentsOf: url.appending(path: "added_tokens.json")),
            let tokens = try? JSONSerialization.jsonObject(with: tokensData) as? [String: Int]
        else { return nil }

        var namesByID: [Int: String] = [:]
        for (name, id) in tokens { namesByID[id] = name }

        for pair in forced where pair.count == 2 {
            guard let name = namesByID[pair[1]],
                  name.hasPrefix("<|"), name.hasSuffix("|>")
            else { continue }
            let code = String(name.dropFirst(2).dropLast(2))
            // Language tags are 2-3 letter codes; the task and control tokens
            // ("transcribe", "notimestamps", …) are longer words.
            guard code.count <= 3, code.allSatisfy(\.isLetter) else { continue }
            return Locale(identifier: code)
        }
        return nil
    }

    /// Whether this export can transcribe `locale`.
    ///
    /// A Whisper export ships pinned to one language, but the pin lives in its
    /// decoder prefix and `CoreAISpeech` can retarget it, so what matters is
    /// whether the bundle's token table knows the language — not what it was
    /// exported with. An export that pins English still transcribes French,
    /// provided `<|fr|>` is in its vocabulary.
    ///
    /// `true` for a bundle with no token table at all (Parakeet), which decodes
    /// multilingual audio directly.
    public func supports(_ locale: Locale) -> Bool {
        guard let code = locale.language.languageCode?.identifier else { return true }
        let tokens = languageTokens
        guard !tokens.isEmpty else { return true }
        return tokens.contains(code.lowercased())
    }

    /// The `<|xx|>` language codes this bundle's token table names.
    ///
    /// Empty when the bundle ships no table, which is how Parakeet appears.
    public var languageTokens: Set<String> {
        guard let url = resolvedURL else { return [] }
        for name in ["added_tokens.json", "vocab.json"] {
            guard let data = try? Data(contentsOf: url.appending(path: name)),
                  let table = try? JSONSerialization.jsonObject(with: data) as? [String: Int]
            else { continue }
            let codes = table.keys.compactMap { token -> String? in
                guard token.hasPrefix("<|"), token.hasSuffix("|>") else { return nil }
                let code = String(token.dropFirst(2).dropLast(2))
                guard code.count >= 2, code.count <= 3, code.allSatisfy(\.isLetter) else { return nil }
                return code.lowercased()
            }
            if !codes.isEmpty { return Set(codes) }
        }
        return []
    }
}
