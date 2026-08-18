// swift-tools-version: 6.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CirceKit",
    platforms: [
        .iOS("27.0"),
        .macOS("27.0"),
        .visionOS("27.0"),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "CirceKit",
            targets: ["CirceKit"]
        ),
    ],
    dependencies: [
        // Fork of apple/coreai-models. Sibling checkout: Circe/ -> development/ -> apple/.
        .package(path: "../../apple/coreai-models.bjnortier"),
    ],
    targets: [
        // whisper.cpp v1.9.1, vendored as a prebuilt XCFramework (module `whisper`).
        .binaryTarget(
            name: "whisper",
            url: "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-v1.9.1-xcframework.zip",
            checksum: "8c3ecbe73f48b0cb9318fc3058264f951ab336fd530e82c4ccdd2298d1311a4c"
        ),
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "CirceKit",
            dependencies: [
                "whisper",
                // Note: for a path dependency the package identity is the *directory* name,
                // not `Package.name` (which is "coreai-models").
                .product(name: "CoreAISpeech", package: "coreai-models.bjnortier"),
            ],
            resources: [
                // British->American spelling map used by EnglishTextNormalizer.
                .copy("Metrics/english.json"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .testTarget(
            name: "CirceKitTests",
            dependencies: ["CirceKit"],
            resources: [
                .copy("Resources"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@loader_path/../../.."])]
        ),
    ],
    swiftLanguageModes: [.v6]
)
