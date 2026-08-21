import CoreAISpeech
import Foundation
import Testing
@testable import CirceKit

/// The compute-unit policy is a mirror of `CoreAISpeech.SpeechComputeUnits`, so
/// the only thing that can go wrong on this side is the mapping — and a mapping
/// that silently collapses to the default would make an A/B on device read as
/// "placement makes no difference" rather than as a bug.
@Suite("Core AI compute units")
struct CoreAIComputeUnitsTests {
    @Test("Every unit maps to its CoreAISpeech counterpart")
    func unitMapping() {
        let expected: [CoreAIComputeUnit: SpeechComputeUnit] = [
            .automatic: .automatic,
            .cpu: .cpu,
            .gpu: .gpu,
            .neuralEngine: .neuralEngine,
        ]
        // Driven off `allCases` so a new unit fails here rather than silently
        // mapping to whatever the switch's other arms happen to cover.
        for unit in CoreAIComputeUnit.allCases {
            #expect(unit.coreAIValue == expected[unit], "\(unit) maps wrong")
        }
    }

    @Test("Both stages survive the mapping independently")
    func stageMapping() {
        let units = CoreAIComputeUnits(encoder: .gpu, decoder: .neuralEngine)
        #expect(units.coreAIValue.encoder == .gpu)
        #expect(units.coreAIValue.decoder == .neuralEngine)
    }

    /// The default has to stay the one `CoreAISpeech` itself ships, or CirceKit
    /// would silently benchmark a different policy than the upstream measurements.
    @Test("The default matches CoreAISpeech's own")
    func defaultMatchesUpstream() {
        #expect(CoreAIComputeUnits.default.coreAIValue == SpeechComputeUnits.default)
        #expect(CoreAIComputeUnits.unspecialized.coreAIValue == SpeechComputeUnits.unspecialized)
    }

    /// The picker persists through `@AppStorage`, which round-trips the value as
    /// a string; a lossy encoding would reset the user's choice on relaunch —
    /// exactly the launch a cold-load comparison depends on.
    @Test("Round-trips through its raw value")
    func rawValueRoundTrip() throws {
        for encoder in CoreAIComputeUnit.allCases {
            for decoder in CoreAIComputeUnit.allCases {
                let units = CoreAIComputeUnits(encoder: encoder, decoder: decoder)
                #expect(CoreAIComputeUnits(rawValue: units.rawValue) == units)
            }
        }
        #expect(CoreAIComputeUnits(rawValue: "nonsense") == nil)
        #expect(CoreAIComputeUnits(rawValue: "gpu") == nil)
    }
}
