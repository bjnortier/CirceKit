import AVFoundation
import CoreMedia
import Foundation
import Speech
import os

/// Apple `SpeechAnalyzer` backend.
///
/// The thinnest of the three: results map field-for-field onto CirceKit's, and
/// the `AttributedString` passes through untouched, so its `.audioTimeRange` and
/// `.transcriptionConfidence` attributes survive intact.
internal final class AppleBackend: TranscriptionBackend {
    private let locale: Locale
    private let transcriptionOptions: Set<CirceTranscriber.TranscriptionOption>
    private let reportingOptions: Set<CirceTranscriber.ReportingOption>
    private let attributeOptions: Set<CirceTranscriber.ResultAttributeOption>

    /// The locale Apple actually supports for ``locale``, resolved once in
    /// ``prepare()``. The `SpeechTranscriber` itself is *not* cached: its results
    /// sequence terminates when a run finalizes, so reusing one across runs would
    /// yield no results the second time. Assets stay warm in the OS, so building a
    /// fresh transcriber per run is cheap.
    private let resolvedLocale = OSAllocatedUnfairLock<Locale?>(initialState: nil)

    init(
        locale: Locale,
        transcriptionOptions: Set<CirceTranscriber.TranscriptionOption>,
        reportingOptions: Set<CirceTranscriber.ReportingOption>,
        attributeOptions: Set<CirceTranscriber.ResultAttributeOption>
    ) {
        self.locale = locale
        self.transcriptionOptions = transcriptionOptions
        self.reportingOptions = reportingOptions
        self.attributeOptions = attributeOptions
    }

    /// Apple negotiates its own format; CirceKit converts to whatever it asks for.
    var analyzerFormat: AVAudioFormat? {
        get async {
            guard let locale = resolvedLocale.withLock({ $0 }) else { return nil }
            return await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [makeTranscriber(locale: locale)])
        }
    }

    /// Apple supports every attribute CirceKit exposes.
    var unsupportedOptions: Set<CirceTranscriber.ResultAttributeOption> { [] }

    func prepare() async throws {
        guard resolvedLocale.withLock({ $0 }) == nil else { return }

        guard let resolved = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
            throw CirceError.localeNotSupported(locale)
        }

        try await Self.ensureModel(for: makeTranscriber(locale: resolved), locale: resolved)
        resolvedLocale.withLock { $0 = resolved }
    }

    /// A transcriber configured with this backend's options.
    private func makeTranscriber(locale: Locale) -> SpeechTranscriber {
        SpeechTranscriber(
            locale: locale,
            transcriptionOptions: Set(transcriptionOptions.map(\.appleValue)),
            reportingOptions: Set(reportingOptions.map(\.appleValue)),
            attributeOptions: Set(attributeOptions.map(\.appleValue))
        )
    }

    func run(
        inputs: AsyncStream<CirceAnalyzerInput>,
        emit: @Sendable @escaping (CirceTranscriber.Result) -> Void
    ) async throws {
        try await prepare()
        guard let locale = resolvedLocale.withLock({ $0 }) else {
            throw CirceError.invalidState("Apple transcriber was not prepared")
        }

        // A new transcriber and analyzer for every run — see `resolvedLocale`.
        let transcriber = makeTranscriber(locale: locale)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        // The results sequence only terminates once the analyzer finishes, so it
        // must be drained concurrently — reading it after driving the analyzer
        // would deadlock.
        let collector = Task {
            for try await result in transcriber.results {
                emit(CirceTranscriber.Result(
                    range: result.range,
                    resultsFinalizationTime: result.resultsFinalizationTime,
                    text: result.text,
                    alternatives: result.alternatives
                ))
            }
        }

        do {
            let converter = format.map { BufferConverter(targetFormat: $0) }
            let appleInputs = AsyncStream<AnalyzerInput> { continuation in
                Task {
                    for await input in inputs {
                        do {
                            let buffer = try converter?.convert(input.buffer) ?? input.buffer
                            // A converted buffer's frame count no longer matches the
                            // source timeline: resampling 44.1 kHz to the analyzer's
                            // rate rounds per chunk, and the accumulated drift makes
                            // the declared start times overlap, which the analyzer
                            // rejects as disordered audio. Hand converted buffers over
                            // without a timestamp and let the analyzer sequence them
                            // contiguously, which is what Apple's own sample does.
                            let startTime = converter == nil ? input.bufferStartTime : nil
                            continuation.yield(
                                AnalyzerInput(buffer: buffer, bufferStartTime: startTime)
                            )
                        } catch {
                            // A buffer that will not convert is dropped rather than
                            // failing the whole run; the analyzer sees a gap.
                            continue
                        }
                    }
                    // Drain the resampler's tail before ending the stream.
                    if let tail = try? converter?.flush() ?? nil {
                        continuation.yield(AnalyzerInput(buffer: tail, bufferStartTime: nil))
                    }
                    continuation.finish()
                }
            }

            // analyzeSequence drives the input to completion; start() would return
            // immediately and finalize before the audio had been consumed.
            if let lastTime = try await analyzer.analyzeSequence(appleInputs) {
                try await analyzer.finalizeAndFinish(through: lastTime)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
        } catch {
            collector.cancel()
            throw error
        }

        try await collector.value
    }

    /// Ensures the locale's assets are installed, downloading them if needed.
    private static func ensureModel(for transcriber: SpeechTranscriber, locale: Locale) async throws {
        let installed = await SpeechTranscriber.installedLocales
        let identifier = locale.identifier(.bcp47)
        if installed.contains(where: { $0.identifier(.bcp47) == identifier }) { return }

        // A nil request means the assets are already present.
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }
    }
}
