#!/bin/sh
# Fetch the real NWB files the tests check against (tests/test_dandi.py) into .dandi-cache/.
# They are public DANDI assets; the tests that need them are skipped until they're here.
set -eu
cd "$(dirname "$0")/.."
mkdir -p .dandi-cache

fetch() { # <file name> <DANDI asset id> <bytes>
    if [ -f ".dandi-cache/$1" ] && [ "$(wc -c < ".dandi-cache/$1")" -eq "$3" ]; then
        echo "have $1"
        return
    fi
    echo "fetching $1"
    curl -fSL --retry 3 -o ".dandi-cache/$1.part" "https://api.dandiarchive.org/api/assets/$2/download/"
    mv ".dandi-cache/$1.part" ".dandi-cache/$1"
}

# DANDI 000714 (Chaudhary / Lu lab NeuroPAL, NWBelegans conventions), sub-8: 49 MB.
fetch 000714_sub-8_ophys.nwb 1433e659-efe7-4abc-a3a8-d384e5686e2f 49172084
