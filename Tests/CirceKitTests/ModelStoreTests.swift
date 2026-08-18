import Foundation
import Testing
@testable import CirceKit

@Suite("Model store", .serialized)
struct ModelStoreTests {
    @Test("The environment override is found without downloading")
    func environmentOverrideIsUsed() throws {
        // Build a fake model directory and check the store finds a file there.
        let directory = URL.temporaryDirectory
            .appending(path: "CirceKitModelStoreTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CirceModelStore(directory: directory)
        #expect(!store.isDownloaded(.tinyEN))

        FileManager.default.createFile(
            atPath: directory.appending(path: WhisperModel.tinyEN.filename).path(percentEncoded: false),
            contents: Data([0x00])
        )
        #expect(store.isDownloaded(.tinyEN))
    }

    @Test("Removing only affects the store's own cache")
    func removeClearsCache() async throws {
        let directory = URL.temporaryDirectory
            .appending(path: "CirceKitModelStoreTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CirceModelStore(directory: directory)
        let path = directory.appending(path: WhisperModel.tinyEN.filename)
        FileManager.default.createFile(atPath: path.path(percentEncoded: false), contents: Data([0x00]))
        #expect(store.isDownloaded(.tinyEN))

        try await store.remove(.tinyEN)
        #expect(!store.isDownloaded(.tinyEN))
    }

    @Test(
        "Downloads and caches a model",
        .enabled(if: TestEnv.allowsNetwork)
    )
    func downloadsAndCaches() async throws {
        let directory = URL.temporaryDirectory
            .appending(path: "CirceKitModelStoreTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { try? FileManager.default.removeItem(at: directory) }

        let store = CirceModelStore(directory: directory)
        let url = try await store.url(for: .tinyEN)

        #expect(FileManager.default.fileExists(atPath: url.path(percentEncoded: false)))
        let size = try FileManager.default
            .attributesOfItem(atPath: url.path(percentEncoded: false))[.size] as? Int ?? 0
        // The real tiny.en model is ~75 MB; anything much smaller is a truncated fetch.
        #expect(size > 50_000_000)

        // Second call must hit the cache, not re-download.
        let again = try await store.url(for: .tinyEN)
        #expect(again == url)
    }
}
