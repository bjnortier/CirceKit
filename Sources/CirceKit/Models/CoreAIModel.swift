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
