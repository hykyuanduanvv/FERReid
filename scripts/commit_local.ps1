# Run from the repository owner's PowerShell. Creates a local commit; never pushes.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    $branch = git branch --show-current
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'main') {
        throw 'Expected the main branch. Review your checkout before committing.'
    }
    $remote = git remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $remote -notin @(
        'https://github.com/hykyuanduanvv/CVPR-2027.git',
        'https://github.com/hykyuanduanvv/CVPR-2027',
        'git@github.com:hykyuanduanvv/CVPR-2027.git'
    )) {
        throw 'Unexpected origin repository.'
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'The index already contains changes. Review and commit them explicitly before running this helper.'
    }
    python scripts/verify_repository.py
    if ($LASTEXITCODE -ne 0) { throw 'Repository verification failed.' }
    git add -- .gitattributes .gitignore README.md THIRD_PARTY_NOTICES.md adapters configs custom_trainer.py data_manifests docs env models.py ops requirements.txt results scripts
    if ($LASTEXITCODE -ne 0) { throw 'git add failed; no commit was created.' }
    git diff --cached --stat
    git commit -m 'Add FERReID experiments, reproducibility artifacts and Chinese deployment docs'
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed; changes remain staged for inspection.' }
    git status --short --branch
    Write-Output 'Local commit created. Nothing was pushed. Review with: git show --stat HEAD'
}
finally {
    Pop-Location
}
