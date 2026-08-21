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
    /// Compute-unit policy for a Core AI backend; ignored by the others.
    public let coreAIComputeUnits: CoreAIComputeUnits

    private let engine: any TranscriptionBackend
    private var isPrepared = false
    /// The language the engine is currently set to, which is the construction
    /// locale until ``transcribe(fileAt:locale:)`` moves it. Tracked rather than
    /// compared against ``locale``, so a transcriber that ran one file in French
    /// goes back to its own locale for the next one instead of staying there.
    private var activeLocale: Locale

    public init(
        backend: CirceTranscriber.Backend,
        locale: Locale = .current,
        preset: CirceTranscriber.Preset = .transcription,
        coreAIComputeUnits: CoreAIComputeUnits = .default
    ) {
        self.init(
            backend: backend,
            locale: locale,
            transcriptionOptions: preset.transcriptionOptions,
            reportingOptions: preset.reportingOptions,
            attributeOptions: preset.attributeOptions,
            coreAIComputeUnits: coreAIComputeUnits
        )
    }

    public init(
        backend: CirceTranscriber.Backend,
        locale: Locale = .current,
        transcriptionOptions: Set<CirceTranscriber.TranscriptionOption> = [],
        reportingOptions: Set<CirceTranscriber.ReportingOption> = [],
        attributeOptions: Set<CirceTranscriber.ResultAttributeOption> = [],
        coreAIComputeUnits: CoreAIComputeUnits = .default
    ) {
        self.backend = backend
        self.locale = locale
        self.activeLocale = locale
        self.attributeOptions = attributeOptions
        self.coreAIComputeUnits = coreAIComputeUnits
        self.engine = backend.makeEngine(
            locale: locale,
            transcriptionOptions: transcriptionOptions,
            reportingOptions: reportingOptions,
            attributeOptions: attributeOptions,
            coreAIComputeUnits: coreAIComputeUnits
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
        try await transcribe(fileAt: url, locale: locale)
    }

    /// Transcribes one file in `locale`, reusing the loaded model.
    ///
    /// A batch engine names the language on each transcription rather than
    /// baking it into the model it loaded, so a corpus that spans languages can
    /// be run through one loaded model — which for a Core AI bundle saves
    /// seconds of load and a gigabyte of resident model per language.
    ///
    /// Throws ``CirceError/invalidState(_:)`` when the engine cannot change
    /// language — Apple's transcriber, or an English-only whisper.cpp model.
    /// Deliberately loud: an engine given the wrong language does not fail, it
    /// transcribes as if the audio were in the language it was told, and scores
    /// like a broken model.
    public func transcribe(fileAt url: URL, locale: Locale) async throws -> CirceTranscription {
        try await prepare()

        if locale.identifier(.bcp47) != activeLocale.identifier(.bcp47) {
            guard engine.retarget(locale: locale) else {
                throw CirceError.invalidState(
                    """
                    This backend cannot change language after it is built — \
                    construct a transcriber for \(locale.identifier(.bcp47)) instead \
                    of retargeting one built for \(self.locale.identifier(.bcp47)).
                    """
                )
            }
            activeLocale = locale
        }

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
