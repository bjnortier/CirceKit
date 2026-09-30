import AVFoundation
import CoreAISpeech
import CoreMedia
import Foundation
import os

/// Decode statistics from a Core AI run, for benchmarking.
public struct CoreAIDecodeStats: Sendable, Equatable {
    /// Number of decoder steps taken.
    public let stepCount: Int
    /// Mean wall-clock time per decoder step.
    public let averageLatencyMs: Double
    /// Decoder steps per second.
    public let stepsPerSecond: Double
    /// How many long-form windows the audio was split into.
    public let windowCount: Int
}

/// Core AI backend, wrapping `CoreAISpeech.SpeechRecognitionModel`.
///
/// Collects the input audio before inference. Can decode a Whisper export with
/// either of Whisper's two tasks — transcribe, or translate to English — through
/// `translatesToEnglish`. With volatile reporting enabled,
/// emits whole-transcript replacement snapshots during Parakeet decoding and
/// after each audio window, followed by one final result. Word timing and
/// confidence attributes remain unsupported.
internal final class CoreAIBackend: TranscriptionBackend {
    private let model: CoreAIModel
    /// The language named on each decode. Mutable because the loaded model is
    /// language-agnostic — see ``retarget(locale:)``.
    private let localeBox: OSAllocatedUnfairLock<Locale>
    private let computeUnits: CoreAIComputeUnits
    private let overlapSeconds: Double
    private let reportsPartials: Bool
    /// Whether to decode with Whisper's `<|translate|>` task instead of `<|transcribe|>`.
    private let translatesToEnglish: Bool
    private let state = OSAllocatedUnfairLock<SpeechRecognitionModel?>(initialState: nil)
    private let statsBox = OSAllocatedUnfairLock<CoreAIDecodeStats?>(initialState: nil)

    init(
        model: CoreAIModel,
        locale: Locale = .current,
        computeUnits: CoreAIComputeUnits = .default,
        overlapSeconds: Double = SpeechRecognitionModel.defaultOverlapSeconds,
        reportsPartials: Bool = false,
        translatesToEnglish: Bool = false
    ) {
        self.model = model
        self.localeBox = OSAllocatedUnfairLock(initialState: locale)
        self.computeUnits = computeUnits
        self.overlapSeconds = overlapSeconds
        self.reportsPartials = reportsPartials
        self.translatesToEnglish = translatesToEnglish
    }

    /// Core AI wants 16 kHz mono float, same as whisper.cpp.
    var analyzerFormat: AVAudioFormat? { AudioDecoder.canonicalFormat }

    /// The engine emits plain text with no timing or confidence.
    var unsupportedOptions: Set<CirceTranscriber.ResultAttributeOption> {
        [.audioTimeRange, .transcriptionConfidence]
    }

    /// Decode statistics from the most recent run.
    var lastStats: CoreAIDecodeStats? { statsBox.withLock { $0 } }

    /// Specializes the graph for `sampleCount` ahead of time.
    ///
    /// Never call this on the path you are timing: it runs a full forward pass,
    /// which roughly doubles the cost of the transcription that follows. It earns
    /// its keep only for a *dynamic* export, where each distinct input length
    /// specializes separately and a benchmark wants that cost outside the clock.
    func prewarm(sampleCount: Int) async throws {
        try await prepare()
        try await state.withLock({ $0 })?.prewarm(sampleCount: sampleCount)
    }

    func prepare() async throws {
        guard state.withLock({ $0 }) == nil else { return }
        let bundleURL = try model.requireURL()
        // Expensive: loads the bundle, specializes the graph, and warms up.
        // Doing it here means it happens once, not per transcription.
        let recognizer = try await SpeechRecognitionModel(
            resourcesAt: bundleURL, computeUnits: computeUnits.coreAIValue
        )
        state.withLock { $0 = recognizer }
    }

    /// The loaded graph is language-agnostic: the language is a decoder prefix
    /// chosen per transcription, so a new one costs nothing.
    func retarget(locale: Locale) -> Bool {
        localeBox.withLock { $0 = locale }
        return true
    }

    func run(
        inputs: AsyncStream<CirceAnalyzerInput>,
        emit: @Sendable @escaping (CirceTranscriber.Result) -> Void
    ) async throws {
        try await prepare()
        guard let recognizer = state.withLock({ $0 }) else {
            throw CirceError.invalidState("Core AI model was not prepared")
        }

        let (samples, duration) = try await AudioLoader.collectPCM16kMono(from: inputs)
        guard !samples.isEmpty else { return }

        // Name the language rather than letting the engine detect it: the caller
        // told us the locale, and detection costs an extra decoder step and can
        // pick wrong on short or noisy audio. Whisper exports transcribe an
        // unnamed language as if it were the one they were pinned to, so getting
        // this wrong is silent. BCP 47 "und" is undetermined, so it asks for detection.
        let code = localeBox.withLock({ $0 }).language.languageCode?.identifier
        let language: SpeechLanguage = code.flatMap { $0 == "und" ? nil : .code($0) } ?? .detect
        let onPartial: (@Sendable (SpeechTranscriptionUpdate) -> Void)?
        if reportsPartials {
            onPartial = { update in
                guard !Task.isCancelled else { return }
                emit(CirceTranscriber.Result(
                    range: CMTimeRange(start: .zero, end: duration),
                    resultsFinalizationTime: .zero,
                    text: AttributedString(update.text),
                    partialSource: update.source == .decoder ? .decoder : .window,
                    progress: update.progress))
            }
        } else {
            onPartial = nil
        }
        // The language stays the *spoken* one when translating: `<|translate|>`
        // goes into the task slot beside it, and Whisper only ever translates into
        // English. Parakeet has no task slot and throws rather than handing back an
        // untranslated transcript.
        let (text, stats) = try await recognizer.transcribe(
            pcm: samples,
            overlapSeconds: overlapSeconds,
            language: language,
            task: translatesToEnglish ? .translateToEnglish : .transcribe,
            onPartial: onPartial
        )

        statsBox.withLock {
            $0 = CoreAIDecodeStats(
                stepCount: stats.stepCount,
                averageLatencyMs: stats.avgLatencyMs,
                stepsPerSecond: stats.stepsPerSecond,
                windowCount: stats.windowCount
            )
        }

        emit(CirceTranscriber.Result(
            range: CMTimeRange(start: .zero, end: duration),
            resultsFinalizationTime: duration,
            text: AttributedString(text.trimmingCharacters(in: .whitespacesAndNewlines)),
            alternatives: [], progress: 1
        ))
    }
}
