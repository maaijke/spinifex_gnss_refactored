# Test Data for spinifex_gnss

This directory contains real GNSS observation and orbit files for testing.

## 📦 Required Files

### RINEX Observation Files (2 files)

1. **WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz** (4.4 MB)
   - Station: WSRT (Westerbork), Netherlands
   - Date: June 17, 2024 (DOY 169)
   - Interval: 30-second observations
   - Format: Hatanaka compressed RINEX 3

2. **WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz** (4.4 MB)
   - Station: WSRT (Westerbork), Netherlands
   - Date: June 18, 2024 (DOY 170)
   - Interval: 30-second observations
   - Format: Hatanaka compressed RINEX 3

### SP3 Orbit Files (3 files)

3. **GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz** (1.0 MB)
   - Source: GFZ Multi-GNSS rapid orbits
   - Date: June 16, 2024 (DOY 168)
   - Interval: 5-minute epochs
   - Satellites: ~140 (GPS, Galileo, GLONASS, BeiDou, QZSS)

4. **GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz** (1.0 MB)
   - Date: June 17, 2024 (DOY 169)

5. **GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz** (1.0 MB)
   - Date: June 18, 2024 (DOY 170)

**Total size:** ~13 MB

---

## 📂 Directory Structure

```
tests/data/
├── DATA_README.md (this file)
├── WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz
├── WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz
├── GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz
├── GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz
└── GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz
```

---

## 🔍 Data Details

### WSRT Station
- **Name:** Westerbork Synthesis Radio Telescope
- **Location:** Netherlands
- **Coordinates:** 52.915°N, 6.603°E
- **Height:** ~16 meters above sea level
- **Network:** IGS (International GNSS Service)

### Observation Period
- **Dates:** June 17-18, 2024
- **Day of Year:** 169-170
- **GPS Week:** 2318

### Coverage
- **RINEX:** 2 days of observations (30-sec intervals)
- **SP3:** 3 days of orbits (5-min intervals)
  - Extra day before/after for interpolation

### Constellations
- **GPS (G):** ~32 satellites
- **Galileo (E):** ~30 satellites
- **GLONASS (R):** ~24 satellites
- **BeiDou (C):** ~50 satellites
- **QZSS (J):** ~4 satellites

---

## 🧪 What Tests Use This Data

### test_integration_real_data.py

**SP3 Parsing Tests:**
- Parse single SP3 file
- Concatenate multiple SP3 files
- Interpolate satellite positions
- Verify coordinate system (IGS20)

**RINEX Parsing Tests:**
- Parse RINEX observation files
- Extract dual-frequency observations
- Select observation codes (C1W, C2W, L1W, L2W)
- Verify data quality

**Integration Tests:**
- Calculate satellite positions
- Calculate satellite-receiver geometry
- Calculate slant distances
- Calculate carrier phase TEC
- Full workflow test

---

## 📥 How to Get These Files

These files were provided earlier in the conversation. To add them to your package:

```bash
# Create data directory
mkdir -p tests/data

# Copy files from wherever you saved them
cp WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz tests/data/
cp WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz tests/data/
cp GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz tests/data/
cp GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz tests/data/
cp GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz tests/data/
```

---

## 🚀 Running Tests with Data

```bash
# Run only integration tests
pytest test_integration_real_data.py -v

# Run all tests including integration
pytest -v

# Skip integration tests (if data not available)
pytest -v -m "not requires_data"
```

---

## ✅ Verify Data Files

```python
from pathlib import Path

data_dir = Path("tests/data")

required_files = [
    "WSRT00NLD_R_20241690000_01D_30S_MO_crx.gz",
    "WSRT00NLD_R_20241700000_01D_30S_MO_crx.gz",
    "GBM0MGXRAP_20241680000_01D_05M_ORB_SP3.gz",
    "GBM0MGXRAP_20241690000_01D_05M_ORB_SP3.gz",
    "GBM0MGXRAP_20241700000_01D_05M_ORB_SP3.gz",
]

for filename in required_files:
    filepath = data_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"✓ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {filename} (missing)")
```

---

## 📊 Expected Test Results

With these files, integration tests should:

### SP3 Tests
- ✓ Parse 3 SP3 files successfully
- ✓ Find ~140 satellites (multi-GNSS)
- ✓ Find ~288 epochs per file (5-min intervals)
- ✓ Verify IGS20 coordinate system
- ✓ Verify GPS week 2318

### RINEX Tests
- ✓ Parse 2 RINEX files successfully
- ✓ Find GPS, Galileo, GLONASS data
- ✓ Extract C1W, C2W, L1W, L2W observations
- ✓ Find ~2880 epochs per file (30-sec intervals)

### Integration Tests
- ✓ Calculate satellite positions within 10 meters
- ✓ Calculate slant distances 15,000-30,000 km
- ✓ Calculate reasonable TEC values (0-100 TECU)
- ✓ Complete workflow without errors

---

## 🔧 File Formats

### RINEX Format
- Version: RINEX 3.x
- Compression: Hatanaka compressed + gzip
- Extension: `.crx.gz`
- Decompression: Automatic (handled by parse_rinex)

### SP3 Format
- Version: SP3-c
- Compression: gzip
- Extension: `.SP3.gz` or `.sp3.gz`
- Decompression: Automatic (handled by parse_sp3)

---

## 💡 Notes

- Files are compressed but read directly (no manual decompression needed)
- SP3 files use 3 days for better interpolation
- RINEX files use 2 consecutive days for 24-hour coverage
- All times are in GPS time system
- Coordinate system is IGS20 (ITRF2020)

---

## 📧 Data Sources

- **RINEX:** IGS data archive
- **SP3:** GFZ (GeoForschungsZentrum Potsdam)
- **Network:** IGS Multi-GNSS

---

**Test data ready to use!** 🎉
