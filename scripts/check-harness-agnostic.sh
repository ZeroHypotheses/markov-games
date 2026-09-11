#!/usr/bin/env bash
# Verify this repo stays harness-agnostic: no harness gets instructions another
# cannot reach, and no instruction exists in two places where it could drift.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

echo "==> Agent-contract files are symlinks to AGENTS.md"
for f in CLAUDE.md GEMINI.md .github/copilot-instructions.md; do
  if [ ! -L "$f" ]; then note FAIL "$f is not a symlink — it is a second copy of the rules"
  elif [ ! -e "$f" ]; then note FAIL "$f is a broken symlink"
  else note ok "$f -> $(readlink "$f")"; fi
done

echo
echo "==> Skills are canonical in .agents/skills, symlinked from .claude/skills"
for d in .agents/skills/*/; do
  n=$(basename "$d")
  [ -f "$d/SKILL.md" ] || note FAIL ".agents/skills/$n has no SKILL.md"
  if [ ! -L ".claude/skills/$n" ]; then
    note FAIL ".claude/skills/$n is a copy, not a symlink — it will drift"
  else note ok "$n"; fi
done

echo
echo "==> Skills stay thin (instructions belong in workflows/)"
for f in .agents/skills/*/SKILL.md; do
  n=$(wc -l < "$f" | tr -d ' ')
  if [ "$n" -gt 40 ]; then
    note FAIL "$f is $n lines — instructions are leaking out of workflows/"
  else note ok "$(basename "$(dirname "$f")") ($n lines)"; fi
  grep -q "workflows/" "$f" || note FAIL "$f does not point at a workflow"
done

echo
echo "==> Every workflow is reachable from AGENTS.md"
for f in workflows/*.md; do
  grep -q "$f" AGENTS.md || note FAIL "$f is not linked from AGENTS.md"
done
[ "$fail" -eq 0 ] && note ok "all $(ls workflows/*.md | wc -l | tr -d ' ') workflows linked"

echo
echo "==> The portable layer carries no harness-specific directives"
# workflows/ is what every harness reads verbatim, so it must name no tool.
# AGENTS.md is the pointer file and may legitimately describe the layout.
if grep -rniE '\.claude/|claude code|skill tool|slash command|(^|[^-a-z])/mg-' workflows/ | grep . ; then
  note FAIL "harness-specific directives found in workflows/ (above)"
else
  note ok "workflows/ names no particular harness"
fi

echo
echo "==> Skills are reachable without a skill mechanism"
for f in .agents/skills/*/SKILL.md; do
  wf=$(grep -oE 'workflows/[a-z-]+\.md' "$f" | head -1)
  [ -f "$wf" ] || note FAIL "$f points at $wf, which does not exist"
done
grep -q "Ask for it by saying" AGENTS.md \
  || note FAIL "AGENTS.md has no natural-language routing table — non-skill harnesses cannot find the workflows"
[ "$fail" -eq 0 ] && note ok "every skill maps to a real workflow, routable in plain words"

echo
echo "==> The rule spec is reachable and referenced"
for f in docs/*.md; do
  grep -q "$f" AGENTS.md || note FAIL "$f is not linked from AGENTS.md"
done

echo
if [ "$fail" -eq 0 ]; then echo "PASS — repo is consistently harness-agnostic."; else
  echo "FAIL — see above."; fi
exit "$fail"
