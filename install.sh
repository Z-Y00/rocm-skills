#!/usr/bin/env bash
# Install the skills and rules as symlinks into this checkout, so `git pull` updates them in place.
#
#   ./install.sh                 # user level: ~/.cursor/skills, ~/.cursor/rules
#   ./install.sh --claude        # also ~/.claude/skills
#   ./install.sh --project DIR   # project level: DIR/.cursor/{skills,rules}, DIR/agent/historical_experience
#   ./install.sh --no-gpulock    # skip the gpu-lock skill
#
# The gpu-lock skill is the gpulock repository itself: cloned to $GPULOCK_HOME (default
# ~/.local/share/gpulock) from $GPULOCK_REPO (default git@github.com:lhzhang333/gpulock.git, HTTPS as a
# fallback; private, so it needs read access), `make install` links ~/.local/bin/gpulock, and the clone is
# linked as the gpu-lock skill. gpulock keeps its clone up to date by itself.
#
# An existing entry that is not a symlink is moved to <target>/.backup-<timestamp>/, never deleted.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
GPULOCK_REPO="${GPULOCK_REPO:-git@github.com:lhzhang333/gpulock.git}"
GPULOCK_HOME="${GPULOCK_HOME:-$HOME/.local/share/gpulock}"
PROJECT=""
CLAUDE=0
GPULOCK=1
while [ $# -gt 0 ]; do
    case "$1" in
        --project) PROJECT="$(cd "$2" && pwd)"; shift 2 ;;
        --claude) CLAUDE=1; shift ;;
        --no-gpulock) GPULOCK=0; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

install_gpulock() {  # clone or update $GPULOCK_HOME; returns non-zero when the repository is out of reach
    if [ -d "$GPULOCK_HOME/.git" ]; then
        git -C "$GPULOCK_HOME" pull -q --ff-only || echo "gpulock: could not update $GPULOCK_HOME; using it as is"
    else
        local https="https://github.com/${GPULOCK_REPO#git@github.com:}"
        mkdir -p "$(dirname "$GPULOCK_HOME")"
        git clone -q "$GPULOCK_REPO" "$GPULOCK_HOME" 2>/dev/null || git clone -q "$https" "$GPULOCK_HOME" || {
            echo "gpulock: cannot clone $GPULOCK_REPO (private: ask for read access); skipping the gpu-lock skill" >&2
            return 1
        }
    fi
    make -s -C "$GPULOCK_HOME" install
    echo "gpulock $("$GPULOCK_HOME/gpulock" --version | cut -d' ' -f2) -> $HOME/.local/bin/gpulock"
}

link() {  # link <source> <target-dir> [<name>]
    local src="$1" dir="$2" dst
    dst="$dir/${3:-$(basename "$src")}"
    mkdir -p "$dir"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        mkdir -p "$dir/.backup-$STAMP"
        mv "$dst" "$dir/.backup-$STAMP/"
        echo "backed up $dst -> $dir/.backup-$STAMP/"
    fi
    ln -sfn "$src" "$dst"
    echo "$dst -> $src"
}

HAVE_GPULOCK=0
[ "$GPULOCK" = 1 ] && install_gpulock && HAVE_GPULOCK=1

install_skills() {  # install_skills <skills-dir>
    for s in "$REPO"/skills/*/; do
        link "${s%/}" "$1"
    done
    [ "$HAVE_GPULOCK" = 1 ] && link "$GPULOCK_HOME" "$1" gpu-lock
    return 0
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
    if [ "$CLAUDE" = 1 ]; then install_skills "$HOME/.claude/skills"; fi
fi
