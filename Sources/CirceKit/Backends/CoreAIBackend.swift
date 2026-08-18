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
/// The least capable of the three engines by interface: it is batch-only, has no
/// locale selection (the language is baked into the export), and produces no
/// timing or confidence information at all. So this backend accumulates the
/// stream, runs one decode, and emits a single final result spanning the whole
/// clip — and reports both attribute options as unsupported rather than
/// pretending otherwise.
internal final class CoreAIBackend: TranscriptionBackend {
    private let model: CoreAIModel
    private let locale: Locale
    private let overlapSeconds: Double
    private let state = OSAllocatedUnfairLock<SpeechRecognitionModel?>(initialState: nil)
    private let statsBox = OSAllocatedUnfairLock<CoreAIDecodeStats?>(initialState: nil)

    init(
        model: CoreAIModel,
        locale: Locale = .current,
        overlapSeconds: Double = SpeechRecognitionModel.defaultOverlapSeconds
    ) {
        self.model = model
        self.locale = locale
        self.overlapSeconds = overlapSeconds
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
        let recognizer = try await SpeechRecognitionModel(resourcesAt: bundleURL, computeUnits: .default)
        state.withLock { $0 = recognizer }
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
        // this wrong is silent.
        let language: SpeechLanguage = locale.language.languageCode
            .map { .code($0.identifier) } ?? .detect
        let (text, stats) = try await recognizer.transcribe(
            pcm: samples,
            overlapSeconds: overlapSeconds,
            language: language
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
            alternatives: []
        ))
    }
}
