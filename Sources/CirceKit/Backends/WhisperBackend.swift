import AVFoundation
import CoreMedia
import Foundation
import Speech
import os
import whisper

/// A single decoded segment as whisper.cpp reports it.
private struct WhisperSegment: Sendable {
    var text: String
    var start: CMTime
    var end: CMTime
    var tokens: [WhisperTokenSpan]
}

/// One token with its timing and probability, used to build attributed runs.
private struct WhisperTokenSpan: Sendable {
    var text: String
    var start: CMTime
    var end: CMTime
    var probability: Double
}

/// Kept alive for the entire synchronous C call. Cancellation can arrive on any thread.
private final class WhisperCallbacks: Sendable {
    let cancelled = OSAllocatedUnfairLock(initialState: false)
    let tokenTimestamps: Bool
    let emit: @Sendable (WhisperSegment) -> Void

    init(tokenTimestamps: Bool, emit: @escaping @Sendable (WhisperSegment) -> Void) {
        self.tokenTimestamps = tokenTimestamps
        self.emit = emit
    }
}

/// Owns a `whisper_context` and serializes access to it.
///
/// The context is not thread-safe, hence the actor. It is held
/// `nonisolated(unsafe)` because `deinit` must free it and a `deinit` cannot be
/// actor-isolated; that is sound here because the actor serializes every other
/// access and `deinit` runs only once the last reference is gone.
private actor WhisperContext {
    private nonisolated(unsafe) let context: OpaquePointer

    init(modelPath: URL) throws {
        var contextParams = whisper_context_default_params()
        contextParams.use_gpu = true
        guard let context = whisper_init_from_file_with_params(
            modelPath.path(percentEncoded: false),
            contextParams
        ) else {
            throw CirceError.modelUnavailable(
                "whisper.cpp could not load the model at \(modelPath.lastPathComponent)"
            )
        }
        self.context = context
    }

    deinit {
        whisper_free(context)
    }

    /// Decodes the input, delivering segments synchronously as whisper produces them.
    func transcribe(
        samples: [Float],
        language: String?,
        callbacks: WhisperCallbacks
    ) throws {
        try Task.checkCancellation()
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.print_realtime = false
        params.print_progress = false
        params.print_timestamps = false
        params.print_special = false
        params.translate = false
        params.no_context = true
        params.single_segment = false
        params.no_timestamps = false
        params.token_timestamps = callbacks.tokenTimestamps
        params.n_threads = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))

        let userData = Unmanaged.passUnretained(callbacks).toOpaque()
        params.abort_callback_user_data = userData
        params.abort_callback = { userData in
            guard let userData else { return false }
            return Unmanaged<WhisperCallbacks>.fromOpaque(userData)
                .takeUnretainedValue().cancelled.withLock { $0 }
        }
        params.new_segment_callback_user_data = userData
        params.new_segment_callback = { context, _, count, userData in
            guard let context, let userData, count > 0 else { return }
            let callbacks = Unmanaged<WhisperCallbacks>.fromOpaque(userData).takeUnretainedValue()
            let total = whisper_full_n_segments(context)
            for index in max(0, total - count)..<total {
                guard !callbacks.cancelled.withLock({ $0 }) else { return }
                callbacks.emit(WhisperContext.segment(
                    context: context, at: index, tokenTimestamps: callbacks.tokenTimestamps))
            }
        }

        // `params.language` is a borrowed `const char *`: whisper_full must run
        // inside the withCString scope or the pointer dangles.
        let languageCode = language ?? "auto"
        let status = withExtendedLifetime(callbacks) {
            languageCode.withCString { languagePointer -> Int32 in
                params.language = languagePointer
                return samples.withUnsafeBufferPointer { buffer in
                    whisper_full(context, params, buffer.baseAddress, Int32(buffer.count))
                }
            }
        }
        if callbacks.cancelled.withLock({ $0 }) { throw CancellationError() }
        try Task.checkCancellation()
        guard status == 0 else {
            throw CirceError.invalidState("whisper_full failed with code \(status)")
        }
    }

    private nonisolated static func segment(context: OpaquePointer, at index: Int32, tokenTimestamps: Bool) -> WhisperSegment {
        let text = whisper_full_get_segment_text(context, index).map { String(cString: $0) } ?? ""
        let segment = WhisperSegment(
            text: text,
            start: Self.time(centiseconds: whisper_full_get_segment_t0(context, index)),
            end: Self.time(centiseconds: whisper_full_get_segment_t1(context, index)),
            tokens: tokenTimestamps ? tokens(context: context, inSegment: index) : []
        )
        return segment
    }

    private nonisolated static func tokens(context: OpaquePointer, inSegment index: Int32) -> [WhisperTokenSpan] {
        let endOfText = whisper_token_eot(context)
        var spans: [WhisperTokenSpan] = []
        for tokenIndex in 0..<whisper_full_n_tokens(context, index) {
            // Special tokens (timestamps, <|endoftext|>, language tags) carry no text.
            guard whisper_full_get_token_id(context, index, tokenIndex) < endOfText else { continue }
            guard let cString = whisper_full_get_token_text(context, index, tokenIndex) else { continue }
            let data = whisper_full_get_token_data(context, index, tokenIndex)
            spans.append(WhisperTokenSpan(
                text: String(cString: cString),
                start: Self.time(centiseconds: data.t0),
                end: Self.time(centiseconds: data.t1),
                probability: Double(data.p)
            ))
        }
        return spans
    }

    /// whisper.cpp reports times in centiseconds.
    private nonisolated static func time(centiseconds: Int64) -> CMTime {
        CMTime(value: CMTimeValue(max(0, centiseconds)), timescale: 100)
    }
}

/// whisper.cpp backend.
///
/// Accumulates the input stream, then emits each segment as decoding produces it. Segment and token timings are real, so
/// `.audioTimeRange` and `.transcriptionConfidence` are both supported.
internal final class WhisperBackend: TranscriptionBackend {
    private let model: WhisperModel
    /// The language passed to `whisper_full`. Mutable because the loaded context
    /// is multilingual — see ``retarget(locale:)``.
    private let localeBox: OSAllocatedUnfairLock<Locale>
    private let attributeOptions: Set<CirceTranscriber.ResultAttributeOption>
    private let store: CirceModelStore

    /// Guarded so the log bridge is installed exactly once per process.
    private static let installLogBridge: Void = {
        whisper_log_set({ level, text, _ in
            guard let text else { return }
            let message = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !message.isEmpty else { return }
            let logger = Logger(subsystem: "CirceKit", category: "whisper.cpp")
            // ggml is chatty at info level; only surface genuine problems.
            if level.rawValue >= GGML_LOG_LEVEL_WARN.rawValue {
                logger.warning("\(message, privacy: .public)")
            } else {
                logger.debug("\(message, privacy: .public)")
            }
        }, nil)
    }()

    private let state = OSAllocatedUnfairLock<WhisperContext?>(initialState: nil)

    init(
        model: WhisperModel,
        locale: Locale,
        attributeOptions: Set<CirceTranscriber.ResultAttributeOption>,
        store: CirceModelStore = .shared
    ) {
        self.model = model
        self.localeBox = OSAllocatedUnfairLock(initialState: locale)
        self.attributeOptions = attributeOptions
        self.store = store
    }

    /// whisper.cpp resamples nothing itself — CirceKit hands it 16 kHz mono float.
    var analyzerFormat: AVAudioFormat? { AudioDecoder.canonicalFormat }

    /// Segment and token timings are both available from whisper.cpp.
    var unsupportedOptions: Set<CirceTranscriber.ResultAttributeOption> { [] }

    func prepare() async throws {
        _ = Self.installLogBridge
        guard state.withLock({ $0 }) == nil else { return }
        let modelURL = try await store.url(for: model)
        let context = try WhisperContext(modelPath: modelURL)
        state.withLock { $0 = context }
    }

    func run(
        inputs: AsyncStream<CirceAnalyzerInput>,
        emit: @Sendable @escaping (CirceTranscriber.Result) -> Void
    ) async throws {
        try await prepare()
        guard let context = state.withLock({ $0 }) else {
            throw CirceError.invalidState("whisper context was not prepared")
        }

        let (samples, duration) = try await AudioLoader.collectPCM16kMono(from: inputs)
        guard !samples.isEmpty else { return }

        let wantsTiming = attributeOptions.contains(.audioTimeRange)
            || attributeOptions.contains(.transcriptionConfidence)
        let callbacks = WhisperCallbacks(tokenTimestamps: wantsTiming) { [self] segment in
            let end = max(segment.end, segment.start)
            emit(CirceTranscriber.Result(
                range: CMTimeRange(start: segment.start, end: end),
                // Decoded segments are final, including ends rounded beyond the clip.
                resultsFinalizationTime: max(duration, end),
                text: attributedText(for: segment),
                alternatives: []
            ))
        }
        try await withTaskCancellationHandler {
            try await context.transcribe(
                samples: samples, language: languageCode, callbacks: callbacks)
        } onCancel: {
            callbacks.cancelled.withLock { $0 = true }
        }
    }

    /// The ISO 639-1 code whisper.cpp expects, or `nil` to auto-detect.
    private var languageCode: String? {
        // English-only models reject anything but "en".
        if model.isEnglishOnly { return "en" }
        return localeBox.withLock({ $0 }).language.languageCode?.identifier
    }

    /// The loaded context is multilingual and takes the language per call, so a
    /// new one costs nothing — except on an English-only model, where anything
    /// but English is refused rather than silently transcribed as English.
    func retarget(locale: Locale) -> Bool {
        guard !model.isEnglishOnly || locale.language.languageCode?.identifier == "en" else {
            return false
        }
        localeBox.withLock { $0 = locale }
        return true
    }

    /// Builds the segment text, attaching per-token time and confidence runs when
    /// they were requested and computed.
    private func attributedText(for segment: WhisperSegment) -> AttributedString {
        let wantsRange = attributeOptions.contains(.audioTimeRange)
        let wantsConfidence = attributeOptions.contains(.transcriptionConfidence)
        guard (wantsRange || wantsConfidence), !segment.tokens.isEmpty else {
            return AttributedString(segment.text)
        }

        var result = AttributedString()
        for token in segment.tokens {
            var run = AttributedString(token.text)
            if wantsRange {
                let end = max(token.end, token.start)
                run.audioTimeRange = CMTimeRange(start: token.start, end: end)
            }
            if wantsConfidence {
                run.transcriptionConfidence = token.probability
            }
            result.append(run)
        }
        return result
    }
}
