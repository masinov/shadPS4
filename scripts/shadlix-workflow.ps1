# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

<#
.SYNOPSIS
Automates the repeatable Shadlix report, build, package, and transfer workflow.

.EXAMPLE
./scripts/shadlix-workflow.ps1 ImportReport

.EXAMPLE
./scripts/shadlix-workflow.ps1 BuildPackage -Tag bda-predication-fix

.EXAMPLE
./scripts/shadlix-workflow.ps1 Send -Archive ./Build/Shadlix-portable-20260815-bda-predication-fix.zip

.NOTES
ImportReport moves the newest shadlix-report*.zip from Downloads into the ignored local-reports
tree. Send performs a network write only when that explicit action is selected.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('Status', 'ImportReport', 'Build', 'Package', 'BuildPackage', 'Send')]
    [string]$Action = 'Status',

    [string]$Tag,
    [string]$SourceArchive,
    [string]$Archive,
    [string]$RuntimeTemplate,
    [string]$BuildDirectory = 'Build/x64-Clang-RelWithDebInfo-Qt',
    [string]$DownloadsDirectory = (Join-Path $env:USERPROFILE 'Downloads'),
    [string]$QtDirectory = 'C:/Qt/6.11.1/msvc2022_64',
    # Transfer target (user@host). Set SHADLIX_REMOTE in the environment or pass -Remote; the
    # default is intentionally not stored in the repository.
    [string]$Remote = $env:SHADLIX_REMOTE,
    [string]$RemoteDirectory = 'Downloads/',
    [ValidateRange(1, 64)]
    [int]$Jobs = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$BuildRoot = Join-Path $RepoRoot 'Build'
$LocalReportsRoot = Join-Path $RepoRoot 'local-reports'
$ResolvedBuildDirectory = [IO.Path]::GetFullPath((Join-Path $RepoRoot $BuildDirectory))
$WorkflowLogRoot = Join-Path $LocalReportsRoot '.workflow-logs'

function Write-CompactResult {
    param([Parameter(Mandatory)]$Value)
    $Value | ConvertTo-Json -Depth 6 -Compress
}

function Get-Sha256 {
    param([Parameter(Mandatory)][string]$Path)
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Resolve-WorkflowPath {
    param([Parameter(Mandatory)][string]$Path)
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Get-UniqueDirectory {
    param([Parameter(Mandatory)][string]$PreferredPath)
    if (-not (Test-Path -LiteralPath $PreferredPath)) {
        return $PreferredPath
    }
    $stamp = Get-Date -Format 'HHmmss'
    $candidate = "$PreferredPath-$stamp"
    $suffix = 1
    while (Test-Path -LiteralPath $candidate) {
        $candidate = "$PreferredPath-$stamp-$suffix"
        ++$suffix
    }
    return $candidate
}

function Resolve-VsDevCmd {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installation) {
            $candidate = Join-Path ($installation | Select-Object -First 1) 'Common7/Tools/VsDevCmd.bat'
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }

    $candidates = Get-ChildItem 'C:/Program Files/Microsoft Visual Studio' -Filter VsDevCmd.bat -File -Recurse -ErrorAction SilentlyContinue
    $match = $candidates | Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $match) {
        throw 'Visual Studio C++ environment was not found (VsDevCmd.bat is missing).'
    }
    return $match.FullName
}

function Resolve-Ninja {
    $command = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    if (Test-Path -LiteralPath 'C:/ninja/ninja.exe') {
        return 'C:/ninja/ninja.exe'
    }
    throw 'Ninja was not found on PATH or at C:/ninja/ninja.exe.'
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Command
    )
    New-Item -ItemType Directory -Path $WorkflowLogRoot -Force | Out-Null
    $logPath = Join-Path $WorkflowLogRoot ("{0}-{1}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $Name)
    $savedErrorPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps any native stderr line in NativeCommandError. CMake uses
        # stderr for non-fatal status output, so the process exit code is the authoritative result.
        $ErrorActionPreference = 'Continue'
        & $env:ComSpec /d /c $Command *> $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorPreference
    }
    if ($exitCode -ne 0) {
        Write-Host "Command failed; final output from $logPath" -ForegroundColor Red
        Get-Content -LiteralPath $logPath -Tail 100
        throw "$Name failed with exit code $exitCode."
    }
    return $logPath
}

function Import-ShadlixReport {
    New-Item -ItemType Directory -Path $LocalReportsRoot -Force | Out-Null

    if ($SourceArchive) {
        $source = Get-Item -LiteralPath (Resolve-WorkflowPath $SourceArchive)
    } else {
        $candidates = @(Get-ChildItem -LiteralPath $DownloadsDirectory -File -Filter 'shadlix-report*.zip' |
            Sort-Object LastWriteTime -Descending)
        if ($candidates.Count -eq 0) {
            throw "No shadlix-report*.zip archive exists in $DownloadsDirectory."
        }
        $source = $candidates[0]
    }

    $reportName = [IO.Path]::GetFileNameWithoutExtension($source.Name)
    $destinationDirectory = Get-UniqueDirectory (Join-Path $LocalReportsRoot $reportName)
    New-Item -ItemType Directory -Path $destinationDirectory | Out-Null
    $storedArchive = Join-Path $destinationDirectory 'report.zip'
    Move-Item -LiteralPath $source.FullName -Destination $storedArchive

    $extractedDirectory = Join-Path $destinationDirectory 'extracted'
    Expand-Archive -LiteralPath $storedArchive -DestinationPath $extractedDirectory
    $files = @(Get-ChildItem -LiteralPath $extractedDirectory -File -Recurse | Sort-Object FullName)
    $manifestFiles = @($files | ForEach-Object {
        [pscustomobject]@{
            Path = $_.FullName.Substring($extractedDirectory.Length).TrimStart('\', '/')
            Bytes = $_.Length
            Modified = $_.LastWriteTime.ToString('o')
        }
    })
    $manifest = [pscustomobject]@{
        ImportedAt = (Get-Date).ToString('o')
        SourceName = $source.Name
        ArchiveSha256 = Get-Sha256 $storedArchive
        Files = $manifestFiles
    }
    $manifestPath = Join-Path $destinationDirectory 'IMPORT-MANIFEST.json'
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    return [pscustomobject]@{
        Action = 'ImportReport'
        ReportDirectory = $destinationDirectory
        Archive = $storedArchive
        ArchiveSha256 = $manifest.ArchiveSha256
        Extracted = $extractedDirectory
        FileCount = $files.Count
        Logs = @($files | Where-Object Extension -EQ '.log' | Sort-Object Length -Descending | ForEach-Object FullName)
        Dumps = @($files | Where-Object Extension -EQ '.dmp' | ForEach-Object FullName)
        Telemetry = @($files | Where-Object Extension -EQ '.csv' | ForEach-Object FullName)
        Manifest = $manifestPath
    }
}

function Build-Shadlix {
    $vsDevCmd = Resolve-VsDevCmd
    $ninja = Resolve-Ninja
    $ninjaDirectory = Split-Path -Parent $ninja
    $env:PATH = "$ninjaDirectory;$env:PATH"
    if (Test-Path -LiteralPath (Join-Path $QtDirectory 'bin')) {
        $env:PATH = "$(Join-Path $QtDirectory 'bin');$env:PATH"
    }

    # Process-local safe-directory configuration handles managed/sandboxed invocations without
    # mutating the user's global Git configuration.
    $env:GIT_CONFIG_COUNT = '1'
    $env:GIT_CONFIG_KEY_0 = 'safe.directory'
    # CMake also queries nested submodule repositories. The wildcard is process-local and expires
    # with this workflow process; it does not weaken the user's persistent Git configuration.
    $env:GIT_CONFIG_VALUE_0 = '*'

    $cmakeArgs = @(
        '-S', ('"{0}"' -f $RepoRoot),
        '-B', ('"{0}"' -f $ResolvedBuildDirectory),
        '-G', 'Ninja',
        '-DENABLE_TESTS=OFF',
        ('-DCMAKE_MAKE_PROGRAM="{0}"' -f $ninja)
    )
    if (Test-Path -LiteralPath $QtDirectory) {
        $cmakeArgs += ('-DCMAKE_PREFIX_PATH="{0}"' -f $QtDirectory)
    }
    $configure = 'call "{0}" -no_logo -arch=x64 -host_arch=x64 && cmake {1}' -f $vsDevCmd, ($cmakeArgs -join ' ')
    $configureLog = Invoke-LoggedCommand -Name 'configure-production' -Command $configure

    $build = 'call "{0}" -no_logo -arch=x64 -host_arch=x64 && cmake --build "{1}" --target shadps4 -j {2}' -f $vsDevCmd, $ResolvedBuildDirectory, $Jobs
    $buildLog = Invoke-LoggedCommand -Name 'build-production' -Command $build

    $exe = Join-Path $ResolvedBuildDirectory 'shadps4.exe'
    $pdb = Join-Path $ResolvedBuildDirectory 'shadps4.pdb'
    if (-not (Test-Path -LiteralPath $exe) -or -not (Test-Path -LiteralPath $pdb)) {
        throw 'Production build completed without both shadps4.exe and shadps4.pdb.'
    }
    return [pscustomobject]@{
        Action = 'Build'
        BuildDirectory = $ResolvedBuildDirectory
        Executable = $exe
        ExecutableSha256 = Get-Sha256 $exe
        Symbols = $pdb
        SymbolsSha256 = Get-Sha256 $pdb
        ConfigureLog = $configureLog
        BuildLog = $buildLog
        TestsEnabled = $false
    }
}

function Resolve-RuntimeTemplate {
    if ($RuntimeTemplate) {
        $resolved = Resolve-WorkflowPath $RuntimeTemplate
        if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
            throw "Runtime template does not exist: $resolved"
        }
        return $resolved
    }

    $templates = @(Get-ChildItem -LiteralPath $BuildRoot -Directory -Filter 'Shadlix-portable*' |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName 'Qt6Core.dll')) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'platforms/qwindows.dll'))
        } | Sort-Object LastWriteTime -Descending)
    if ($templates.Count -eq 0) {
        throw 'No existing portable runtime directory was found. Pass -RuntimeTemplate explicitly.'
    }
    return $templates[0].FullName
}

function Package-Shadlix {
    if ([string]::IsNullOrWhiteSpace($Tag)) {
        throw 'Package and BuildPackage require -Tag (for example: -Tag bda-predication-fix).'
    }
    $safeTag = $Tag.ToLowerInvariant() -replace '[^a-z0-9-]+', '-'
    $safeTag = $safeTag.Trim('-')
    if (-not $safeTag) {
        throw 'Tag contains no usable characters.'
    }

    $exe = Join-Path $ResolvedBuildDirectory 'shadps4.exe'
    $pdb = Join-Path $ResolvedBuildDirectory 'shadps4.pdb'
    if (-not (Test-Path -LiteralPath $exe) -or -not (Test-Path -LiteralPath $pdb)) {
        throw 'Build output is missing. Run the Build action first.'
    }

    $template = Resolve-RuntimeTemplate
    $baseName = 'Shadlix-portable-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd'), $safeTag
    $packageDirectory = Join-Path $BuildRoot $baseName
    $archivePath = "$packageDirectory.zip"
    if ((Test-Path -LiteralPath $packageDirectory) -or (Test-Path -LiteralPath $archivePath)) {
        throw "Package already exists: $baseName. Choose a new -Tag; existing artifacts are never overwritten."
    }

    Copy-Item -LiteralPath $template -Destination $packageDirectory -Recurse
    Copy-Item -LiteralPath $exe -Destination (Join-Path $packageDirectory 'shadps4.exe') -Force
    Copy-Item -LiteralPath $pdb -Destination (Join-Path $packageDirectory 'shadps4.pdb') -Force

    $packageFiles = @(Get-ChildItem -LiteralPath $packageDirectory -File -Recurse)
    $unexpected = @($packageFiles | Where-Object {
        $_.FullName.Substring($packageDirectory.Length) -match '(?i)[\\/](user|shader_cache|logs?|reports?|games?)([\\/]|$)'
    })
    if ($unexpected.Count -ne 0) {
        throw "Portable package contains unexpected user-data paths: $($unexpected.FullName -join ', ')"
    }

    Compress-Archive -Path (Join-Path $packageDirectory '*') -DestinationPath $archivePath -CompressionLevel Optimal
    $exeHash = Get-Sha256 (Join-Path $packageDirectory 'shadps4.exe')
    $buildExeHash = Get-Sha256 $exe
    if ($exeHash -ne $buildExeHash) {
        throw 'Packaged executable does not match the production build.'
    }

    return [pscustomobject]@{
        Action = 'Package'
        Template = $template
        PackageDirectory = $packageDirectory
        Archive = $archivePath
        ArchiveBytes = (Get-Item -LiteralPath $archivePath).Length
        ArchiveSha256 = Get-Sha256 $archivePath
        ExecutableSha256 = $exeHash
        SymbolsSha256 = Get-Sha256 (Join-Path $packageDirectory 'shadps4.pdb')
        EntryCount = $packageFiles.Count
        UnexpectedUserDataEntries = $unexpected.Count
        ScpCommand = 'scp "{0}" {1}:{2}' -f $archivePath, $(if ($Remote) { $Remote } else { '<user@host>' }), $RemoteDirectory
    }
}

function Send-Shadlix {
    if ([string]::IsNullOrWhiteSpace($Archive)) {
        $candidate = Get-ChildItem -LiteralPath $BuildRoot -File -Filter 'Shadlix-portable-*.zip' |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $candidate) {
            throw 'No portable archive exists. Pass -Archive or package a build first.'
        }
        $archiveItem = $candidate
    } else {
        $archiveItem = Get-Item -LiteralPath (Resolve-WorkflowPath $Archive)
    }
    if ([string]::IsNullOrWhiteSpace($Remote)) {
        throw 'No transfer target: set SHADLIX_REMOTE (user@host) or pass -Remote.'
    }
    & scp.exe $archiveItem.FullName "${Remote}:$RemoteDirectory"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE."
    }
    return [pscustomobject]@{
        Action = 'Send'
        Archive = $archiveItem.FullName
        ArchiveSha256 = Get-Sha256 $archiveItem.FullName
        Destination = "${Remote}:$RemoteDirectory"
    }
}

function Get-WorkflowStatus {
    $downloadReports = @()
    if (Test-Path -LiteralPath $DownloadsDirectory) {
        $downloadReports = @(Get-ChildItem -LiteralPath $DownloadsDirectory -File -Filter 'shadlix-report*.zip' |
            Sort-Object LastWriteTime -Descending | ForEach-Object FullName)
    }
    $localReports = @()
    if (Test-Path -LiteralPath $LocalReportsRoot) {
        $localReports = @(Get-ChildItem -LiteralPath $LocalReportsRoot -Directory |
            Where-Object Name -NotLike '.*' | Sort-Object LastWriteTime -Descending |
            Select-Object -First 10 | ForEach-Object FullName)
    }
    $packages = @()
    if (Test-Path -LiteralPath $BuildRoot) {
        $packages = @(Get-ChildItem -LiteralPath $BuildRoot -File -Filter 'Shadlix-portable-*.zip' |
            Sort-Object LastWriteTime -Descending | Select-Object -First 10 | ForEach-Object FullName)
    }
    $exe = Join-Path $ResolvedBuildDirectory 'shadps4.exe'
    return [pscustomobject]@{
        Action = 'Status'
        Repository = $RepoRoot
        DownloadsReports = $downloadReports
        LocalReports = $localReports
        BuildExecutable = if (Test-Path -LiteralPath $exe) { $exe } else { $null }
        BuildExecutableSha256 = if (Test-Path -LiteralPath $exe) { Get-Sha256 $exe } else { $null }
        PortableArchives = $packages
    }
}

Push-Location $RepoRoot
try {
    $result = switch ($Action) {
        'Status' { Get-WorkflowStatus }
        'ImportReport' { Import-ShadlixReport }
        'Build' { Build-Shadlix }
        'Package' { Package-Shadlix }
        'BuildPackage' {
            $buildResult = Build-Shadlix
            $packageResult = Package-Shadlix
            [pscustomobject]@{ Action = 'BuildPackage'; Build = $buildResult; Package = $packageResult }
        }
        'Send' { Send-Shadlix }
    }
    Write-CompactResult $result
} finally {
    Pop-Location
}
