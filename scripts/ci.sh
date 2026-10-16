#!/usr/bin/env bash
# Runs a CI command. On failure, shows its last lines as an error on the PR page.
# NOTICE=1: also show them as a note when it passes (e.g. audit summaries).
log=$(mktemp)
"$@" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
msg=$(tail -n "${TAIL:-40}" "$log" | tr -d '\r' | sed 's/\x1b\[[0-9;]*m//g; s/%/%25/g' | sed ':a;N;$!ba;s/\n/%0A/g')
if [ "$code" -ne 0 ]; then
  echo "::error title=Failed: $*::$msg"
elif [ "${NOTICE:-0}" = 1 ]; then
  echo "::notice title=$*::$msg"
fi
exit "$code"
