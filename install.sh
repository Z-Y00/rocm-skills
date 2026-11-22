#!/usr/bin/env bash
# Install the skills and rules as symlinks into this checkout, so `git pull` updates them in place.
#
#   ./install.sh                 # user level: ~/.cursor/skills, ~/.cursor/rules
#   ./install.sh --claude        # also ~/.claude/skills
#   ./install.sh --project DIR   # project level: DIR/.cursor/{skills,rules}, DIR/agent/historical_experience
#
# An existing entry that is not a symlink is moved to <target>/.backup-<timestamp>/, never deleted.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
PROJECT=""
CLAUDE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --project) PROJECT="$(cd "$2" && pwd)"; shift 2 ;;
        --claude) CLAUDE=1; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

link() {  # link <source> <target-dir>
    local src="$1" dir="$2" dst
    dst="$dir/$(basename "$src")"
    mkdir -p "$dir"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        mkdir -p "$dir/.backup-$STAMP"
        mv "$dst" "$dir/.backup-$STAMP/"
        echo "backed up $dst -> $dir/.backup-$STAMP/"
    fi
    ln -sfn "$src" "$dst"
    echo "$dst -> $src"
}

install_skills() {  # install_skills <skills-dir>
    for s in "$REPO"/skills/*/; do
        link "${s%/}" "$1"
    done
}

if [ -n "$PROJECT" ]; then
    install_skills "$PROJECT/.cursor/skills"
    for r in "$REPO"/rules/*.mdc; do link "$r" "$PROJECT/.cursor/rules"; done
    if [ -e "$PROJECT/agent/historical_experience" ] && [ ! -L "$PROJECT/agent/historical_experience" ]; then
        echo "kept $PROJECT/agent/historical_experience (a real directory); merge tips into $REPO/historical_experience by hand"
    else
        link "$REPO/historical_experience" "$PROJECT/agent"
    fi
else
    install_skills "$HOME/.cursor/skills"
    for r in "$REPO"/rules/*.mdc; do link "$r" "$HOME/.cursor/rules"; done
    [ "$CLAUDE" = 1 ] && install_skills "$HOME/.claude/skills"
fi
