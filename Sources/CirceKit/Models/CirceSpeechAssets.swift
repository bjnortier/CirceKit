import Foundation
import Speech

/// Manages the on-device speech assets Apple's transcriber needs.
///
/// Apple's models are installed by the OS rather than fetched by the app, behind
/// a reservation system: a device keeps only a limited number of locales, so a
/// slot must be reserved before installing and released if the install fails.
/// ``CirceTranscriber`` installs assets implicitly when it prepares, which is
/// enough for a one-off transcription; this is for apps that want to manage and
/// show progress for downloads up front.
public enum CirceSpeechAssets {
    /// Locales Apple's transcriber can handle.
    public static var supportedLocales: [Locale] {
        get async { await SpeechTranscriber.supportedLocales }
    }

    /// Locales whose assets are installed on this device.
    public static var installedLocales: [Locale] {
        get async { await SpeechTranscriber.installedLocales }
    }

    /// The supported locale equivalent to `locale`, if there is one.
    public static func supportedLocale(equivalentTo locale: Locale) async -> Locale? {
        await SpeechTranscriber.supportedLocale(equivalentTo: locale)
    }

    /// Whether `locale`'s assets are already installed.
    public static func isInstalled(_ locale: Locale) async -> Bool {
        let identifier = locale.identifier(.bcp47)
        return await installedLocales.contains { $0.identifier(.bcp47) == identifier }
    }

    /// How many locales this device will keep reserved at once.
    public static var maximumReservedLocales: Int {
        AssetInventory.maximumReservedLocales
    }

    /// Locales currently holding a reservation slot.
    public static var reservedLocales: [Locale] {
        get async { await AssetInventory.reservedLocales }
    }

    /// Claims a reservation slot for `locale`.
    @discardableResult
    public static func reserve(locale: Locale) async throws -> Bool {
        try await AssetInventory.reserve(locale: locale)
    }

    /// Gives back `locale`'s reservation slot.
    @discardableResult
    public static func release(locale: Locale) async -> Bool {
        await AssetInventory.release(reservedLocale: locale)
    }

    /// Releases every reservation this app holds.
    public static func releaseAllReservations() async {
        for locale in await reservedLocales {
            _ = await release(locale: locale)
        }
    }

    /// Downloads and installs `locale`'s assets, reporting fractional progress.
    ///
    /// Reserves a slot first and releases it again if anything fails, so a failed
    /// install does not leak one of the device's limited reservations. Returns
    /// immediately when the assets are already present.
    public static func install(
        locale: Locale,
        progress progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        guard let resolved = await supportedLocale(equivalentTo: locale) else {
            throw CirceError.localeNotSupported(locale)
        }
        if await isInstalled(resolved) {
            progressHandler?(1.0)
            return
        }

        var didReserve = false
        do {
            didReserve = try await reserve(locale: resolved)
            guard didReserve else {
                throw CirceError.modelUnavailable(
                    """
                    Reservation limit reached — this device keeps at most \
                    \(maximumReservedLocales) speech locales installed.
                    """
                )
            }

            let transcriber = SpeechTranscriber(
                locale: resolved,
                transcriptionOptions: [],
                reportingOptions: [],
                attributeOptions: []
            )
            // A nil request means the assets arrived between the check and here.
            guard let request = try await AssetInventory.assetInstallationRequest(
                supporting: [transcriber]
            ) else {
                progressHandler?(1.0)
                return
            }

            // The request reports progress through a Progress object rather than a
            // callback, so poll it alongside the install.
            let poller = progressHandler.map { handler in
                let progress = request.progress
                return Task {
                    while !Task.isCancelled, !progress.isFinished {
                        handler(progress.fractionCompleted)
                        try? await Task.sleep(for: .milliseconds(250))
                    }
                }
            }
            defer { poller?.cancel() }

            try await request.downloadAndInstall()
            progressHandler?(1.0)
        } catch {
            if didReserve { _ = await release(locale: resolved) }
            throw error
        }
    }
}
