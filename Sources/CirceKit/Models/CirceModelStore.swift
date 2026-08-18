import Foundation

/// Downloads and caches whisper.cpp ggml models.
///
/// Resolution order for ``url(for:progress:)``:
/// 1. `CIRCEKIT_MODEL_DIR`, if set and the file is there — nothing is downloaded.
/// 2. The store's own cache directory, if the file has already been fetched.
/// 3. A fresh download from HuggingFace.
public actor CirceModelStore {
    /// Environment variable naming a directory of pre-existing ggml models.
    public static let modelDirectoryEnvironmentKey = "CIRCEKIT_MODEL_DIR"

    public static let shared = CirceModelStore()

    /// Where downloads are cached.
    public nonisolated let directory: URL

    /// In-flight downloads, so concurrent requests for one model share a fetch.
    private var inFlight: [WhisperModel: Task<URL, any Error>] = [:]

    public init(directory: URL? = nil) {
        if let directory {
            self.directory = directory
        } else {
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
                ?? URL.temporaryDirectory
            self.directory = caches.appending(path: "CirceKit/Models", directoryHint: .isDirectory)
        }
    }

    /// The externally-provided model directory, if `CIRCEKIT_MODEL_DIR` names one.
    private nonisolated var overrideDirectory: URL? {
        guard let path = ProcessInfo.processInfo.environment[Self.modelDirectoryEnvironmentKey],
              !path.isEmpty
        else { return nil }
        return URL(filePath: path, directoryHint: .isDirectory)
    }

    /// A local file URL for `model`, downloading it if necessary.
    ///
    /// `progress` receives fractions in `0...1`. It is called from the download
    /// delegate, not from the caller's context.
    public func url(
        for model: WhisperModel,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        if let existing = locate(model) { return existing }

        // Coalesce concurrent requests for the same model onto one download.
        if let running = inFlight[model] { return try await running.value }

        let task = Task<URL, any Error> { [directory] in
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appending(path: model.filename)
            try await Self.download(from: model.remoteURL, to: destination, progress: progress)
            return destination
        }
        inFlight[model] = task
        defer { inFlight[model] = nil }
        return try await task.value
    }

    /// Whether `model` is already available locally, from either directory.
    public nonisolated func isDownloaded(_ model: WhisperModel) -> Bool {
        locate(model) != nil
    }

    /// Deletes `model` from the store's cache. Never touches `CIRCEKIT_MODEL_DIR`,
    /// which holds files the store did not create.
    public func remove(_ model: WhisperModel) throws {
        let cached = directory.appending(path: model.filename)
        if FileManager.default.fileExists(atPath: cached.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: cached)
        }
    }

    /// Finds an existing copy of `model` without downloading.
    private nonisolated func locate(_ model: WhisperModel) -> URL? {
        for base in [overrideDirectory, directory].compacted() {
            let candidate = base.appending(path: model.filename)
            if FileManager.default.fileExists(atPath: candidate.path(percentEncoded: false)) {
                return candidate
            }
        }
        return nil
    }

    /// Downloads `url` to `destination`, moving into place only on success so a
    /// failed or cancelled fetch never leaves a truncated model behind.
    private static func download(
        from url: URL,
        to destination: URL,
        progress: (@Sendable (Double) -> Void)?
    ) async throws {
        let (temporary, response) = try await URLSession.shared.download(from: url)

        // The temporary file is deleted when this scope exits unless it is moved.
        var moved = false
        defer { if !moved { try? FileManager.default.removeItem(at: temporary) } }

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw CirceError.modelUnavailable(
                "HTTP \(http.statusCode) fetching \(url.lastPathComponent)"
            )
        }

        if FileManager.default.fileExists(atPath: destination.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
        moved = true
        progress?(1.0)
    }
}

private extension Sequence {
    /// `compactMap { $0 }` for a sequence of optionals, without the closure.
    func compacted<Wrapped>() -> [Wrapped] where Element == Wrapped? {
        compactMap { $0 }
    }
}
