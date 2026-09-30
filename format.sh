#!/bin/sh
find Sources Tests -name '*.swift' | xargs xcrun swift-format -i
