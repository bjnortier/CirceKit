import AVFoundation
import CoreMedia
import Foundation
import Speech
import os
import whisper

/// Transcribes speech to text, over a choice of ASR engine.
///
/// Mirrors `Speech.SpeechTranscriber`. The only structural addition is
/// ``Backend``: everything else — presets, option sets, the shape of
/// ``Result`` — matches Apple's, so porting code between the two is a rename.
public final class CirceTranscriber: CirceSpeechModule {
    /// Which ASR engine does the work.
    public enum Backend: Sendable, Equatable, Hashable {
        /// Apple's on-device `SpeechTranscriber`.
        case apple
        /// A Core AI export (Parakeet TDT or Whisper), via `CoreAISpeech`.
        case coreAI(CoreAIModel)
        /// A whisper.cpp ggml model.
        case whisperCPP(WhisperModel)
    }

    // MARK: Options

    public enum TranscriptionOption: Sendable, Equatable, Hashable, CaseIterable {
        case etiquetteReplacements

        var appleValue: SpeechTranscriber.TranscriptionOption {
            switch self {
            case .etiquetteReplacements: return .etiquetteReplacements
            }
        }
    }

    public enum ReportingOption: Sendable, Equatable, Hashable, CaseIterable {
        case volatileResults
        case alternativeTranscriptions
        case fastResults

        var appleValue: SpeechTranscriber.ReportingOption {
            switch self {
            case .volatileResults: return .volatileResults
            case .alternativeTranscriptions: return .alternativeTranscriptions
            case .fastResults: return .fastResults
            }
        }
    }

    public enum ResultAttributeOption: Sendable, Equatable, Hashable, CaseIterable {
        case audioTimeRange
        case transcriptionConfidence

        var appleValue: SpeechTranscriber.ResultAttributeOption {
            switch self {
            case .audioTimeRange: return .audioTimeRange
            case .transcriptionConfidence: return .transcriptionConfidence
            }
        }
    }

    /// A named bundle of the three option sets, mirroring Apple's presets.
    public struct Preset: Sendable, Equatable, Hashable {
        public var transcriptionOptions: Set<TranscriptionOption>
        public var reportingOptions: Set<ReportingOption>
        public var attributeOptions: Set<ResultAttributeOption>

        public init(
            transcriptionOptions: Set<TranscriptionOption>,
            reportingOptions: Set<ReportingOption>,
            attributeOptions: Set<ResultAttributeOption>
        ) {
            self.transcriptionOptions = transcriptionOptions
            self.reportingOptions = reportingOptions
            self.attributeOptions = attributeOptions
        }

        /// Plain final transcription.
        public static let transcription = Preset(
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: []
        )

        /// Final transcription with per-word time ranges.
        public static let timeIndexedTranscription = Preset(
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange]
        )

        /// Volatile results as they arrive, then finals.
        public static let progressiveTranscription = Preset(
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: []
        )

        /// Volatile results plus per-word time ranges.
        public static let timeIndexedProgressiveTranscription = Preset(
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange]
        )
    }

    // MARK: Result

    /// A span of transcribed audio.
    ///
    /// ``text`` carries Apple's `SpeechAttributes` — `.audioTimeRange` and
    /// `.transcriptionConfidence` — regardless of which engine produced it, so a
    /// whisper.cpp result is indistinguishable in shape from an Apple one.
    public struct Result: CirceSpeechModuleResult, Sendable, Equatable, CustomStringConvertible {
        public let range: CMTimeRange
        public let resultsFinalizationTime: CMTime
        public let text: AttributedString
        public let alternatives: [AttributedString]
        public enum PartialSource: Sendable { case decoder, window }
        /// Whole-transcript replacement snapshot; nil for ordinary engine results.
        public let partialSource: PartialSource?
        /// Fraction of source audio decoded; nil when the backend cannot measure it.
        public let progress: Double?

        public init(
            range: CMTimeRange,
            resultsFinalizationTime: CMTime,
            text: AttributedString,
            alternatives: [AttributedString] = [],
            partialSource: PartialSource? = nil,
            progress: Double? = nil
        ) {
            self.range = range
            self.resultsFinalizationTime = resultsFinalizationTime
            self.text = text
            self.alternatives = alternatives
            self.partialSource = partialSource
            self.progress = progress
        }

        public var description: String {
            let marker = isFinal ? "final" : "volatile"
            return "[\(marker) \(range.start.seconds)-\(range.end.seconds)] \(String(text.characters))"
        }
    }

    // MARK: State

    public let backend: Backend
    public let locale: Locale
    public let transcriptionOptions: Set<TranscriptionOption>
    public let reportingOptions: Set<ReportingOption>
    public let attributeOptions: Set<ResultAttributeOption>
    /// Compute-unit policy for a ``Backend/coreAI(_:)`` backend; ignored by the others.
    public let coreAIComputeUnits: CoreAIComputeUnits
    /// Translate the audio into English instead of transcribing it as spoken.
    ///
    /// A Core AI **Whisper** export only: the task is one token of Whisper's
    /// decoder prefix, and Whisper's translation task has English as its only
    /// target — ``locale`` still names the language being *spoken*. Every other
    /// backend rejects this at ``prepare()`` rather than quietly transcribing;
    /// see ``Backend/canTranslateToEnglish``.
    public let translatesToEnglish: Bool

    private let engine: any TranscriptionBackend
    private let stream: AsyncThrowingStream<Result, any Error>
    private let continuation: AsyncThrowingStream<Result, any Error>.Continuation
    private let runState = OSAllocatedUnfairLock<Task<Void, any Error>?>(initialState: nil)

    public convenience init(
        backend: Backend,
        locale: Locale = .current,
        preset: Preset = .transcription,
        coreAIComputeUnits: CoreAIComputeUnits = .default,
        translatesToEnglish: Bool = false
    ) {
        self.init(
            backend: backend,
            locale: locale,
            transcriptionOptions: preset.transcriptionOptions,
            reportingOptions: preset.reportingOptions,
            attributeOptions: preset.attributeOptions,
            coreAIComputeUnits: coreAIComputeUnits,
            translatesToEnglish: translatesToEnglish
        )
    }

    public init(
        backend: Backend,
        locale: Locale = .current,
        transcriptionOptions: Set<TranscriptionOption> = [],
        reportingOptions: Set<ReportingOption> = [],
        attributeOptions: Set<ResultAttributeOption> = [],
        coreAIComputeUnits: CoreAIComputeUnits = .default,
        translatesToEnglish: Bool = false
    ) {
        self.backend = backend
        self.locale = locale
        self.transcriptionOptions = transcriptionOptions
        self.reportingOptions = reportingOptions
        self.attributeOptions = attributeOptions
        self.coreAIComputeUnits = coreAIComputeUnits
        self.translatesToEnglish = translatesToEnglish

        engine = backend.makeEngine(
            locale: locale,
            transcriptionOptions: transcriptionOptions,
            reportingOptions: reportingOptions,
            attributeOptions: attributeOptions,
            coreAIComputeUnits: coreAIComputeUnits,
            translatesToEnglish: translatesToEnglish
        )

        (stream, continuation) = AsyncThrowingStream<Result, any Error>.makeStream()
    }

    /// Results, delivered as the engine produces them. Finishes when the run ends.
    public var results: AsyncThrowingStream<Result, any Error> { stream }

    /// Audio formats this transcriber can consume directly.
    public var availableCompatibleAudioFormats: [AVAudioFormat] {
        get async {
            if let format = await engine.analyzerFormat { return [format] }
            return [AudioDecoder.canonicalFormat]
        }
    }

    /// Requested attribute options the selected engine cannot deliver.
    ///
    /// Asking Core AI for `.audioTimeRange` is a known gap, for instance — this
    /// reports it instead of silently returning results without timings.
    public var unsupportedOptions: Set<ResultAttributeOption> {
        attributeOptions.intersection(engine.unsupportedOptions)
    }

    /// Loads models and warms up, so the first transcription is not also a download.
    public func prepare() async throws {
        try backend.validateTranslation(translatesToEnglish)
        try await engine.prepare()
    }

    /// Core AI decode statistics from the most recent run, when that is the backend.
    public var coreAIStats: CoreAIDecodeStats? {
        (engine as? CoreAIBackend)?.lastStats
    }

    // MARK: Availability

    /// Locales the backend can transcribe.
    public static func supportedLocales(for backend: Backend) async -> [Locale] {
        switch backend {
        case .apple:
            return await SpeechTranscriber.supportedLocales
        case .whisperCPP(let model):
            // whisper.cpp multilingual models cover every language it knows;
            // the `.en` variants are English-only.
            return model.isEnglishOnly ? [Locale(identifier: "en")] : whisperSupportedLocales()
        case .coreAI:
            // The language is fixed at export time and not queryable.
            return []
        }
    }

    /// Locales whose assets are present locally.
    public static func installedLocales(for backend: Backend) async -> [Locale] {
        switch backend {
        case .apple:
            return await SpeechTranscriber.installedLocales
        case .whisperCPP(let model):
            return CirceModelStore.shared.isDownloaded(model)
                ? await supportedLocales(for: backend)
                : []
        case .coreAI:
            // The language is fixed at export time, so there is nothing to report.
            return []
        }
    }

    /// Whether the backend can run right now, without downloading anything.
    public static func isAvailable(_ backend: Backend) async -> Bool {
        switch backend {
        case .apple:
            return SpeechTranscriber.isAvailable
        case .whisperCPP(let model):
            return CirceModelStore.shared.isDownloaded(model)
        case .coreAI(let model):
            return model.resolvedURL != nil
        }
    }

    /// The languages whisper.cpp ships with.
    private static func whisperSupportedLocales() -> [Locale] {
        (0...whisperMaxLanguageID).compactMap { id in
            whisper_lang_str(Int32(id)).map { Locale(identifier: String(cString: $0)) }
        }
    }

    private static var whisperMaxLanguageID: Int { Int(whisper_lang_max_id()) }
}

// MARK: - AnalyzerAttachable

extension CirceTranscriber: AnalyzerAttachable {
    var analyzerFormat: AVAudioFormat? {
        get async { await engine.analyzerFormat }
    }

    func attach(inputs: AsyncStream<CirceAnalyzerInput>) async throws {
        guard runState.withLock({ $0 }) == nil else {
            throw CirceError.invalidState("this transcriber is already running")
        }
        // Checked here as well as in `prepare()`: a run started through the analyzer
        // never calls `prepare()` itself, and a request to translate that reached the
        // engine unhonoured would come back as a fluent transcript in the wrong language.
        try backend.validateTranslation(translatesToEnglish)
        let continuation = self.continuation
        let engine = self.engine
        let task = Task {
            do {
                try await engine.run(inputs: inputs) { result in
                    continuation.yield(result)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
                throw error
            }
        }
        runState.withLock { $0 = task }
    }

    func finishAndFinalize() async throws {
        guard let task = runState.withLock({ $0 }) else { return }
        // The run ends when its input stream ends; wait for the results to drain.
        try await task.value
    }

    func cancelNow() async {
        runState.withLock { $0 }?.cancel()
        continuation.finish()
    }
}

// MARK: - Engine construction

extension CirceTranscriber.Backend {
    /// Whether this backend can translate its audio into English.
    ///
    /// Core AI only, and there only for a Whisper export — the task is a token of
    /// Whisper's decoder prefix. Parakeet has no such slot, and neither Apple's
    /// transcriber nor this package's whisper.cpp backend exposes one. Whether a
    /// given Core AI bundle is Whisper is not knowable until it is loaded, so that
    /// half of the check belongs to `CoreAISpeech`, which throws on the decode.
    public var canTranslateToEnglish: Bool {
        if case .coreAI = self { return true }
        return false
    }

    /// Throws unless `translatesToEnglish` is something this backend can honour.
    ///
    /// Loud rather than ignored, for the same reason ``TranscriptionBackend/retarget(locale:)``
    /// reports failure: a translation request that is silently dropped comes back
    /// as a fluent transcript in the source language, which reads as a working run.
    internal func validateTranslation(_ translatesToEnglish: Bool) throws {
        guard translatesToEnglish, !canTranslateToEnglish else { return }
        throw CirceError.invalidState(
            """
            This backend cannot translate to English — that is Whisper's \
            `<|translate|>` decoder task, which only the Core AI backend exposes. \
            Transcribe with this backend, or choose a Core AI Whisper export.
            """
        )
    }

    /// Builds the engine that implements this backend.
    ///
    /// Shared by ``CirceTranscriber`` and ``CirceFileTranscriber`` so the two
    /// entry points can never drift on how a backend is configured.
    internal func makeEngine(
        locale: Locale,
        transcriptionOptions: Set<CirceTranscriber.TranscriptionOption>,
        reportingOptions: Set<CirceTranscriber.ReportingOption>,
        attributeOptions: Set<CirceTranscriber.ResultAttributeOption>,
        coreAIComputeUnits: CoreAIComputeUnits,
        translatesToEnglish: Bool = false
    ) -> any TranscriptionBackend {
        switch self {
        case .apple:
            AppleBackend(
                locale: locale,
                transcriptionOptions: transcriptionOptions,
                reportingOptions: reportingOptions,
                attributeOptions: attributeOptions
            )
        case .coreAI(let model):
            CoreAIBackend(
                model: model, locale: locale, computeUnits: coreAIComputeUnits,
                reportsPartials: reportingOptions.contains(.volatileResults),
                translatesToEnglish: translatesToEnglish)
        case .whisperCPP(let model):
            WhisperBackend(
                model: model,
                locale: locale,
                attributeOptions: attributeOptions
            )
        }
    }
}
