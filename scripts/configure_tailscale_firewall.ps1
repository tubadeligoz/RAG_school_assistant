$ErrorActionPreference = "Stop"

$ruleName = "DORA Streamlit over Tailscale 8501"
$existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if (-not $existingRule) {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 8501 `
        -RemoteAddress 100.64.0.0/10 `
        -Profile Any | Out-Null
}

Get-NetFirewallRule -DisplayName $ruleName |
    Select-Object DisplayName, Enabled, Direction, Action
