import AVFoundation
import CoreMedia
import Foundation

/// What every ASR engine must provide for ``CirceTranscriber`` to drive it.
///
/// The contract is deliberately *stream in, results out* even though two of the
/// three engines are batch-only. Apple's transcriber is genuinely incremental,
/// and modelling to the most capable engine means the batch backends degrade
/// gracefully (accumulate, run once, emit finals) rather than the streaming one
/// being flattened to a single blocking call.
internal protocol TranscriptionBackend: AnyObject, Sendable {
    /// The audio format this engine wants, once known. `nil` means "anything
    /// decodable" — the batch engines resample internally.
    var analyzerFormat: AVAudioFormat? { get async }

    /// Loads models and warms up. Called once before ``run(inputs:emit:)``.
    func prepare() async throws

    /// Consumes `inputs` to completion, calling `emit` for each result.
    ///
    /// Returns only when the input stream has ended and all results have been
    /// emitted. Options the engine cannot honour are reported through
    /// ``TranscriptionBackend/unsupportedOptions``, not silently dropped.
    func run(
        inputs: AsyncStream<CirceAnalyzerInput>,
        emit: @Sendable @escaping (CirceTranscriber.Result) -> Void
    ) async throws

    /// Requested options this engine cannot deliver.
    var unsupportedOptions: Set<CirceTranscriber.ResultAttributeOption> { get }
}
