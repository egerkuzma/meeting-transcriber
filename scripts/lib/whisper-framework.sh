#!/usr/bin/env bash
# Single source of truth for getting whisper.framework into an app bundle.
#
# Unlike every other binary dependency in this project, the whisper.cpp
# xcframework ships a *dynamic* framework: the app binary records
# `@rpath/whisper.framework/Versions/Current/whisper` and dyld refuses to launch
# without it. SwiftPM copies the framework next to the product in
# `.build/<config>/`, which is why `swift build` / `swift test` / `swift run`
# work with no help — but both bundle assemblers copy only the executable, so
# the bundle they produce is missing it. Measured, not theorised: adding the
# binary target to the target's dependencies (before a single Swift file
# imported it) was already enough to make the assembled bundle die at launch
# with "Library not loaded".
#
# Destination is Contents/Frameworks, the documented place for nested
# frameworks, reached through the `@executable_path/../Frameworks` rpath that
# Package.swift adds at link time. Both are needed: the rpath without the copy
# resolves to nothing, and the copy without the rpath is never searched.
#
# Source this, don't execute it.

# Copies the framework SwiftPM resolved for the given build configuration into
# `<bundle>/Contents/Frameworks`. Fails loudly rather than leaving a bundle that
# looks assembled and dies at launch.
#
# Every step propagates its own failure explicitly instead of relying on the
# caller's `set -e`, for the reason spelled out in localvqe-resources.sh: one
# caller invokes this from an `if !` condition, which suppresses errexit
# recursively.
#
# $1: the .app bundle
# $2: the SwiftPM package directory (contains .build)
# $3: the build configuration directory name (e.g. "release")
install_whisper_framework() {
    local app_bundle="$1" spm_dir="$2" config="${3:-release}"
    local source_framework dest_dir

    [ -n "$app_bundle" ] || { echo "  ERROR: no app bundle given" >&2; return 1; }
    [ -n "$spm_dir" ] || { echo "  ERROR: no SwiftPM directory given" >&2; return 1; }

    source_framework="$spm_dir/.build/$config/whisper.framework"
    if [ ! -d "$source_framework" ]; then
        echo "  ERROR: whisper.framework not found at $source_framework" >&2
        echo "         (expected SwiftPM to have staged it next to the built product)" >&2
        return 1
    fi

    dest_dir="$app_bundle/Contents/Frameworks"
    mkdir -p "$dest_dir" || return 1
    # --delete so a stale framework from an earlier xcframework pin cannot
    # survive alongside the new one; -a to keep the symlinks a versioned macOS
    # framework is built from (Versions/Current, and the top-level aliases),
    # which a plain `cp -R` would flatten into duplicate copies and codesign
    # would then reject as a malformed bundle.
    rsync -a --delete "$source_framework" "$dest_dir/" || return 1

    echo "  whisper.framework: $dest_dir/whisper.framework"
}
