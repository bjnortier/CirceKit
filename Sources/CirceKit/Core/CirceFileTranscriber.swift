import AVFoundation
import CoreMedia
import Foundation
import os

/// The outcome of transcribing one file.
public struct CirceTranscription: Sendable {
    /// The final results, concatenated and trimmed.
    public let text: String

    /// Every result the engine produced, in emission order.
    public let results: [CirceTranscriber.Result]

    /// How much audio the file held.
    public let audioDuration: CMTime

    public init(text: String, results: [CirceTranscriber.Result], audioDuration: CMTime) {
        self.text = text
        self.results = results
        self.audioDuration = audioDuration
    }
}

/// Transcribes whole files with a model that stays loaded between calls.
///
/// ``CirceTranscriber`` mirrors `Speech.SpeechTranscriber` and is therefore
/// single-use: its results sequence ends when its run does. That is the right
/// shape for streaming, and the wrong shape for batch work — transcribing a
/// corpus one file at a time would reload the model per file, and Core AI's
/// graph specialization alone takes seconds.
///
/// This type keeps one prepared engine and runs it repeatedly, so model loading
/// is paid once. Use it for benchmarking, bulk transcription, and anywhere the
/// audio is already a complete file.
public actor CirceFileTranscriber {
    public let backend: CirceTranscriber.Backend
    public let locale: Locale
    public let attributeOptions: Set<CirceTranscriber.ResultAttributeOption>

    private let engine: any TranscriptionBackend
    private var isPrepared = false

    public init(
        backend: CirceTranscriber.Backend,
        locale: Locale = .current,
        preset: CirceTranscriber.Preset = .transcription
    ) {
        self.init(
            backend: backend,
            locale: locale,
            transcriptionOptions: preset.transcriptionOptions,
            reportingOptions: preset.reportingOptions,
            attributeOptions: preset.attributeOptions
        )
    }

    public init(
        backend: CirceTranscriber.Backend,
        locale: Locale = .current,
        transcriptionOptions: Set<CirceTranscriber.TranscriptionOption> = [],
        reportingOptions: Set<CirceTranscriber.ReportingOption> = [],
        attributeOptions: Set<CirceTranscriber.ResultAttributeOption> = []
    ) {
        self.backend = backend
        self.locale = locale
        self.attributeOptions = attributeOptions
        self.engine = backend.makeEngine(
            locale: locale,
            transcriptionOptions: transcriptionOptions,
            reportingOptions: reportingOptions,
            attributeOptions: attributeOptions
        )
    }

    /// Loads the model and warms it up.
    ///
    /// Idempotent, and called automatically by ``transcribe(fileAt:)``. Call it
    /// explicitly before a batch so the first file is not also a model load.
    public func prepare() async throws {
        guard !isPrepared else { return }
        try await engine.prepare()
        isPrepared = true
    }

    /// Transcribes one file using the already-loaded model.
    ///
    /// Safe to call repeatedly; the actor serializes runs.
    public func transcribe(fileAt url: URL) async throws -> CirceTranscription {
        try await prepare()

        let file = try AVAudioFile(forReading: url)
        let duration = AudioLoader.duration(of: file)
        let inputs = try AudioLoader.inputStream(from: file)

        let collected = OSAllocatedUnfairLock<[CirceTranscriber.Result]>(initialState: [])
        try await engine.run(inputs: inputs) { result in
            collected.withLock { $0.append(result) }
        }

        let results = collected.withLock { $0 }
        // Volatile results are superseded by the finals that follow them.
        let text = results
            .filter(\.isFinal)
            .map { String($0.text.characters) }
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return CirceTranscription(text: text, results: results, audioDuration: duration)
    }

    /// Requested attribute options the selected engine cannot deliver.
    public var unsupportedOptions: Set<CirceTranscriber.ResultAttributeOption> {
        attributeOptions.intersection(engine.unsupportedOptions)
    }

    /// Core AI decode statistics from the most recent run, when that is the backend.
    public var coreAIStats: CoreAIDecodeStats? {
        (engine as? CoreAIBackend)?.lastStats
    }

    /// The audio format the engine consumes.
    public var analyzerFormat: AVAudioFormat? {
        get async { await engine.analyzerFormat }
    }
}
