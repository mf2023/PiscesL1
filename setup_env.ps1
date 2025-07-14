# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

Write-Host "🐟 Setting up Pisces environment..." -ForegroundColor Green

# Check if Python 3 is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv pisces_env

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& ".\pisces_env\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Environment setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
    Write-Host ".\pisces_env\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "To deactivate, run:" -ForegroundColor Cyan
    Write-Host "deactivate" -ForegroundColor White
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
} 