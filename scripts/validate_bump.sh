#!/bin/bash
set -euo pipefail

echo
echo "Validating bump"
echo "==============="

echoerr() {
    >&2 echo "$@"
}

extract_version() {
    # Remove comments and spaces from input (file or string).
    if [[ -f "$1" ]]; then
        content="$(cat "$1")"
    else
        content="$1"
    fi
    echo "${content}" | grep -o '^[^#]*' | tr -d "[:space:]"
}

get_default_branch() {
    local ref="refs/remotes/origin/"
    git symbolic-ref "${ref}HEAD" | sed "s@^${ref}@@"
    # Alternative is "git remote show origin", but it takes ~2 s
}

parent_branch="$(get_default_branch)"

version_new=$(uvx --python 3.11 python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
version_old=$(uvx --python 3.11 python -c "import tomllib; print(tomllib.loads('''$(git show origin/${parent_branch}:pyproject.toml)''')['project']['version'])" || echo "")

version_new_str="$(extract_version "${version_new}")"
version_old_str="$(extract_version "${version_old}")" || {
    version_old_str=""
}

echo "Old version = \"${version_old_str}\""
echo "New version = \"${version_new_str}\""
echo "Parent branch = \"${parent_branch}\""

version_regex="^[0-9]+\.[0-9]+\.[0-9]+$"

# Adapted from https://peps.python.org/pep-0440/#public-version-identifiers:
version_regex_general="^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?$"

if ! grep -Eq "${version_regex_general}" <<<"$version_new_str"; then
    echoerr "Version must be of the form X.Y.Z[{a|b|rc}N][.postM][.devL], where X, Y, Z, N, M, L are integers, but found: \"${version_new_str}\""
    echo "Reference: https://peps.python.org/pep-0440/#public-version-identifiers"
    exit 1
fi

if ! grep -Eq "${version_regex}" <<<"$version_new_str"; then
    echo "New version not of the form X.Y.Z. Skipping version check."
    exit 0
fi

if [[ -z $version_old_str ]] || [[ $parent_branch != "master" && $parent_branch != "main" ]] || {
    ! grep -Eq "${version_regex}" <<<"$version_old_str"
}; then
    echo "Test skipped"
    exit 0
fi

if [[ "${version_old_str}" != "${version_new_str}" ]]; then
    # Version changed

    IFS='.' read -r -a version_new <<<"$version_new_str"
    IFS='.' read -r -a version_old <<<"$version_old_str"

    # Try to find version in the CHANGELOG
    # Patterns: "X.Y.Z", "##... X.Y.Z", "##... [X.Y.Z"
    if [[ -f "CHANGELOG.md" ]]; then
        version_regex="^(#+ \[?)?${version_new_str//./\\.}\b"
        version_changelog=$(grep -E "${version_regex}" "CHANGELOG.md") || :
    fi
    version_changelog="${version_changelog:-}"

    changelog_error_msg="Version \"${version_new_str}\" not found in the CHANGELOG.md."

    # Assuming version is "X.Y.Z"

    # "X" was increased
    if [[ "${version_new[0]}" -gt "${version_old[0]}" ]]; then
        if [[ "${version_new[1]}" -ne 0 || "${version_new[2]}" -ne 0 ]]; then
            echoerr "New version must be of the form \"X.0.0\" (found \"${version_new_str}\")."
            exit 1
        fi
        if [[ -z "${version_changelog}" ]]; then
            echoerr "${changelog_error_msg}"
            exit 1
        fi

    # "Y" was increased (and "X" is the same)
    elif [[ "${version_new[0]}" -eq "${version_old[0]}" &&
            "${version_new[1]}" -gt "${version_old[1]}" ]]; then
        if [[ "${version_new[2]}" -ne 0 ]]; then
            echoerr "New version must be of the form \"X.Y.0\" (found \"${version_new_str}\")."
            exit 1
        fi
        if [[ -z "${version_changelog}" ]]; then
            echoerr "${changelog_error_msg}"
            exit 1
        fi

    # "Z" was increased (and "X" and "Y" are the same)
    elif [[ "${version_new[0]}" -eq "${version_old[0]}" &&
            "${version_new[1]}" -eq "${version_old[1]}" &&
            "${version_new[2]}" -gt "${version_old[2]}" ]]; then
        if [[ -z "${version_changelog}" ]]; then
            echoerr "${changelog_error_msg}"
            #exit 1 # optional
        fi

    # Version was decreased (shouldn't happen)
    else
        echoerr "You cannot decrease the version."
        exit 1
    fi

else
    # Version is the same

    echo
    echo "Modified files"
    echo "=============="
    files_changed=$(git diff --name-only "origin/${parent_branch}...")
    echo "$files_changed"

    files_require_bump_regex="
    src/.*\.py
    uv\.lock
    pyproject\.toml"

    files_require_bump_regex_no_indent=$(echo "$files_require_bump_regex" | sed 's/^[[:space:]]*// ; /^$/d')

    if files_require_bump=$(grep "$files_require_bump_regex_no_indent" <<<"$files_changed"); then
        echo >&2
        echoerr "Version bump is required because the following files were modified:"
        echo >&2 "$files_require_bump"
        echo >&2
        echoerr "Bump the version in \"pyproject.toml\"."
        exit 1
    fi
fi

echo "Version bump and CHANGELOG are OK!"
