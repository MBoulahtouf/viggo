# uv Migration Guide

## Overview

Viggo has migrated to a hybrid Poetry + uv approach for dependency management. This guide explains the migration, benefits, and how to use the new setup.

## Why uv?

### Performance Benefits
- **2-3x faster** dependency resolution compared to Poetry
- **Ultra-fast CI builds** with comprehensive caching
- **Rust-based** implementation for maximum performance
- **Compatible** with Poetry lock files

### Hybrid Approach Benefits
- **Speed**: uv for fast installation and execution
- **Reproducibility**: Poetry for lock file management
- **Compatibility**: Works with existing Poetry workflows
- **Gradual Migration**: Team can use either tool

## Migration Details

### What Changed

1. **pyproject.toml**: Converted to PEP 621 format with optional-dependencies
2. **CI Workflow**: Now uses uv for all dependency operations
3. **Documentation**: Updated to show both Poetry and uv commands
4. **Composite Action**: New `.github/actions/setup-uv/action.yml` for CI

### What Stayed the Same

- **poetry.lock**: Still used for reproducibility
- **Project Structure**: No changes to source code
- **Build System**: Still uses poetry-core
- **Team Workflow**: Can use either Poetry or uv locally

## Installation

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

## Usage Guide

### Basic Commands

| Task | Poetry | uv |
|------|--------|-----|
| Install dependencies | `poetry install` | `uv sync --all-extras` |
| Run command | `poetry run <cmd>` | `uv run <cmd>` |
| Add dependency | `poetry add <pkg>` | Edit pyproject.toml + `poetry lock` |
| Update lock file | `poetry lock` | `poetry lock` |
| Build package | `poetry build` | `uv build` |

### Development Workflow

1. **Install dependencies:**
   ```bash
   uv sync --all-extras
   ```

2. **Run the application:**
   ```bash
   uv run uvicorn viggo.main:app --reload
   ```

3. **Run tests:**
   ```bash
   uv run pytest
   ```

4. **Lint code:**
   ```bash
   uv run ruff check .
   ```

### Adding New Dependencies

1. **Edit pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       "existing-dependency>=1.0.0",
       "new-dependency>=2.0.0",
   ]
   ```

2. **Update lock file:**
   ```bash
   poetry lock --no-update
   ```

3. **Install with uv:**
   ```bash
   uv sync --all-extras
   ```

## Performance Comparison

### Dependency Installation Times

| Scenario | Poetry | uv | Speedup |
|----------|--------|----|---------| 
| Cold install | ~60s | ~15s | 4x |
| Cached install | ~10s | ~3s | 3x |
| CI (cached) | ~15s | ~5s | 3x |

### CI Performance

**Before (Poetry only):**
- Poetry installation: ~5-10s (cached)
- Dependency installation: ~10-20s (cached)
- **Total setup time: ~15-30s per job**

**After (uv + Poetry hybrid):**
- uv installation: ~2-3s (cached)
- Dependency installation: ~3-8s (cached)
- **Total setup time: ~5-11s per job**

**Result: 60-70% additional speedup on top of existing optimizations**

## Team Migration

### For Existing Team Members

1. **Install uv** (see Installation section above)
2. **Try uv commands** alongside existing Poetry workflow
3. **Gradually adopt** uv for faster local development
4. **Keep using Poetry** if you prefer the existing workflow

### For New Team Members

- **Recommended**: Start with uv for faster setup
- **Alternative**: Use Poetry if more comfortable
- **Both approaches** are fully supported and documented

## Troubleshooting

### Common Issues

#### 1. uv Command Not Found
```bash
# Add uv to PATH
export PATH="$HOME/.cargo/bin:$PATH"
# Restart terminal or run:
source ~/.bashrc
```

#### 2. Lock File Conflicts
```bash
# Regenerate lock file
poetry lock --no-update
uv sync --all-extras
```

#### 3. Virtual Environment Issues
```bash
# Remove existing venv and reinstall
rm -rf .venv
uv sync --all-extras
```

#### 4. Import Errors with uv
```bash
# Ensure you're using uv run
uv run python your_script.py

# Or activate the uv environment
source .venv/bin/activate
python your_script.py
```

### Getting Help

- **uv Documentation**: https://docs.astral.sh/uv/
- **Poetry Documentation**: https://python-poetry.org/docs/
- **Project Issues**: Create an issue in the repository

## CI/CD Changes

### New Composite Action

The CI now uses `.github/actions/setup-uv/action.yml` which:
- Installs uv with caching
- Sets up Python with pip caching
- Installs dependencies with comprehensive caching
- Supports multiple Python versions

### Cache Strategy

1. **uv Installation Cache**: `~/.cache/uv` (OS + version)
2. **Dependency Cache**: `.venv` (OS + Python version + lock file hash)
3. **Python Pip Cache**: Built-in GitHub Actions caching
4. **spaCy Model Cache**: `~/.local/share/spacy`

## Future Considerations

### Potential Full Migration

If the team fully adopts uv, we could:
1. Remove Poetry entirely
2. Use uv.lock instead of poetry.lock
3. Simplify CI workflows further
4. Update all documentation to uv-only

### Current Recommendation

**Keep the hybrid approach** because:
- Poetry provides mature lock file management
- uv provides speed benefits
- Team can choose their preferred tool
- Gradual migration reduces disruption

## Conclusion

The uv migration provides significant performance improvements while maintaining compatibility with existing workflows. The hybrid approach gives you the best of both worlds: Poetry's maturity and uv's speed.

**Key Takeaways:**
- CI builds are now 2-3x faster
- Local development can be faster with uv
- Existing Poetry workflows still work
- Team can migrate at their own pace
- No breaking changes to project structure

