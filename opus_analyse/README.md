# Opus Analyse Map

Deze map bevat de uitgebreide deep analysis van de FTMO trading bot, uitgevoerd door Claude Sonnet 4.5 (Opus-niveau analyse).

## 📁 Inhoud

### Hoofd Analyse Document
- **[BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md)** - Comprehensive 15-25 pagina technical deep dive
  - Executive Summary met kritieke bevindingen
  - Baseline performance review (Trial #0 metrics)
  - Complete architectuur analyse (15-flag confluence, ADX regime)
  - Parameter space mapping (19 current, 30+ hardcoded, 14 disabled)
  - Performance deep dive (maandelijks, per symbol)
  - Geprioriteerde improvement roadmap (P0-P3)
  - Code quality assessment
  - Performance projecties (Phase 1-3)
  - Appendices (parameter catalogus, priority matrix, code references)

## 🎯 Doel

Deze analyse is bedoeld als:
1. **Foundation document** voor project manager (Claude Sonnet 4.5)
2. **Input voor Stap 2 prompt**: Strategy Performance Deep Dive
3. **Input voor Stap 3 prompt**: Creative Parameter Design
4. **Implementation roadmap** met gefaseerde aanpak

## 📊 Kritieke Bevindingen

**Baseline Performance (Trial #0):**
- Score: 66.04
- Training: +99.88R (1,517 trades, 48.6% WR)
- Validation: +93.74R (1,018 trades, 49.7% WR)
- **Max DD: 25.9%** ⚠️ FAILS FTMO 10% LIMIT

**Top 3 Critical Issues:**
1. ⚠️ Drawdown exceeds FTMO limit by 15.9 percentage points
2. ⚠️ 12+ trading filters currently DISABLED (minimal signal filtering)
3. ⚠️ Q3 seasonality problem (-80R July-September losses)

**Top 3 P0 Priorities:**
1. Implement drawdown protection system (DD: 25.9% → <10%)
2. Enable core trading filters as optimizable toggles (WR: 48.6% → 55%+)
3. Add Q3 seasonality filter (eliminate -80R seasonal losses)

## 🗺️ Implementation Roadmap

### Phase 1: Critical Fixes (1 Week)
**Goal**: FTMO-compliant system with <10% max drawdown
- Drawdown protection (daily limits, circuit breakers)
- Enable 14 disabled filters
- Q3 seasonality handling
- **Expected**: DD 25.9% → <10%, WR 48.6% → 52%+

### Phase 2: Performance Enhancements (2 Weeks)
**Goal**: Industry-standard metrics (Sharpe >1.5, WR >55%)
- Dynamic TP system
- Symbol-specific parameters
- Correlation limits
- **Expected**: DD <8%, WR >55%, Sharpe >1.5

### Phase 3: Advanced Features (1 Month)
**Goal**: Institutional-grade system with ML
- Random Forest regime classification
- News event filtering
- Portfolio optimization
- **Expected**: DD <5%, Sharpe >2.0, FTMO Pass Rate 80%+

## 📈 Performance Targets

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| Max DD | 25.9% ❌ | <10% ✅ | <8% ✅ | <5% ✅ |
| Win Rate | 48.6% | 52%+ | 55%+ | 58%+ |
| Sharpe | 0.916 | 1.2+ | 1.5+ | 2.0+ |
| FTMO Compliant | NO ❌ | YES ✅ | YES ✅ | YES ✅ |

---

**Analyse Datum**: 2025-12-28  
**Analyst**: Claude Sonnet 4.5  
**Basis Data**: Trial #0 (1-trial optimization run)  
**Data Bron**: `ftmo_analysis_output/TPE/history/run_001/`
