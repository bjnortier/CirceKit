import AVFoundation
import CoreMedia
import Foundation

/// Drives one or more ``CirceSpeechModule``s over a stream of audio.
///
/// Mirrors `Speech.SpeechAnalyzer`. This round implements the file-driven paths;
/// the `inputSequence` overloads exist because reading a file *is* an input
/// sequence, so they come for free.
public actor CirceAnalyzer {
    /// Analysis options, mirroring `SpeechAnalyzer.Options`.
    public struct Options: Sendable, Equatable {
        public let priority: TaskPriority

        public init(priority: TaskPriority = .userInitiated) {
            self.priority = priority
        }
    }

    public nonisolated let modules: [any CirceSpeechModule]
    public nonisolated let options: Options

    /// The modules, downcast once to the interface the analyzer actually drives.
    private let attachables: [any AnalyzerAttachable]
    private var isRunning = false
    private var pumpTask: Task<Void, Never>?
    /// End time of the last input seen, which is what `analyzeSequence` reports.
    private var lastInputEndTime: CMTime?

    public init(modules: [any CirceSpeechModule], options: Options? = nil) {
        self.modules = modules
        self.options = options ?? Options()
        self.attachables = modules.compactMap { $0 as? any AnalyzerAttachable }
    }

    /// Creates an analyzer already reading `inputAudioFile`.
    public init(
        inputAudioFile: AVAudioFile,
        modules: [any CirceSpeechModule],
        options: Options? = nil,
        finishAfterFile: Bool = false
    ) async throws {
        self.init(modules: modules, options: options)
        try await start(inputAudioFile: inputAudioFile, finishAfterFile: finishAfterFile)
    }

    // MARK: Starting

    /// Begins analysis of `inputSequence`, returning once analysis has started.
    public func start<InputSequence>(inputSequence: InputSequence) async throws
    where InputSequence: AsyncSequence & Sendable, InputSequence.Element == CirceAnalyzerInput {
        try validateModules()
        guard !isRunning else {
            throw CirceError.invalidState("this analyzer is already running")
        }
        isRunning = true

        // Fan the one input sequence out to every module.
        var streams: [AsyncStream<CirceAnalyzerInput>] = []
        var continuations: [AsyncStream<CirceAnalyzerInput>.Continuation] = []
        for _ in attachables {
            let (stream, continuation) = AsyncStream<CirceAnalyzerInput>.makeStream()
            streams.append(stream)
            continuations.append(continuation)
        }

        for (module, stream) in zip(attachables, streams) {
            try await module.attach(inputs: stream)
        }

        // Pump the source into every module's stream, tracking how far we got.
        lastInputEndTime = nil
        let pump = Task(priority: options.priority) {
            do {
                for try await input in inputSequence {
                    for continuation in continuations { continuation.yield(input) }
                    if let start = input.bufferStartTime {
                        lastInputEndTime = start + input.bufferDuration
                    } else {
                        lastInputEndTime = (lastInputEndTime ?? .zero) + input.bufferDuration
                    }
                }
            } catch {
                // The source failed; modules see a truncated stream and finalize
                // whatever they have.
            }
            for continuation in continuations { continuation.finish() }
        }
        pumpTask = pump
    }

    /// Begins analysis of `audioFile`.
    public func start(inputAudioFile audioFile: AVAudioFile, finishAfterFile: Bool = false) async throws {
        let inputs = try AudioLoader.inputStream(from: audioFile)
        try await start(inputSequence: inputs)
        if finishAfterFile {
            try await finalizeAndFinishThroughEndOfInput()
        }
    }

    // MARK: Analyzing to completion

    /// Analyzes `inputSequence` to its end, returning the last input's end time.
    ///
    /// Does not finalize — call ``finalizeAndFinishThroughEndOfInput()`` after.
    @discardableResult
    public func analyzeSequence<InputSequence>(_ inputSequence: InputSequence) async throws -> CMTime?
    where InputSequence: AsyncSequence & Sendable, InputSequence.Element == CirceAnalyzerInput {
        try await start(inputSequence: inputSequence)
        await pumpTask?.value
        return lastInputEndTime
    }

    /// Analyzes `audioFile` to its end, returning the file's end time.
    @discardableResult
    public func analyzeSequence(from audioFile: AVAudioFile) async throws -> CMTime? {
        let inputs = try AudioLoader.inputStream(from: audioFile)
        let duration = AudioLoader.duration(of: audioFile)
        try await start(inputSequence: inputs)
        await pumpTask?.value
        lastInputEndTime = duration
        return duration
    }

    // MARK: Finishing

    /// Finalizes all results and ends every module's results sequence.
    public func finalizeAndFinishThroughEndOfInput() async throws {
        await pumpTask?.value
        for module in attachables {
            try await module.finishAndFinalize()
        }
        isRunning = false
    }

    /// Abandons analysis without finalizing.
    public func cancelAndFinishNow() async {
        pumpTask?.cancel()
        for module in attachables {
            await module.cancelNow()
        }
        isRunning = false
    }

    // MARK: Formats

    /// The best audio format every module in `modules` can consume.
    public static func bestAvailableAudioFormat(
        compatibleWith modules: [any CirceSpeechModule]
    ) async -> AVAudioFormat? {
        var formats: [AVAudioFormat]?
        for module in modules {
            let available = await module.availableCompatibleAudioFormats
            formats = formats.map { $0.filter(available.contains) } ?? available
        }
        return formats?.first
    }

    /// Rejects modules the analyzer has no way to drive, rather than silently
    /// producing no results for them.
    private func validateModules() throws {
        for module in modules where !(module is any AnalyzerAttachable) {
            throw CirceError.unsupportedModule(String(describing: type(of: module)))
        }
    }
}
