import { useEffect, useRef, useState } from 'react'

import { fetchApiHealth } from './api/health.js'
import {
  createValuation,
  fetchModelStatus,
  fetchRecentPredictions,
} from './api/valuation.js'

const STATUS_COPY = { checking: 'Connecting to API', online: 'API online', offline: 'API offline' }
const INITIAL_FORM = {
  year: '2020', make: 'Toyota', model: 'Camry', status: 'used', mileage: '48000', coverage: '0.9',
}
const EXAMPLES = [
  { label: 'Sedan · Toyota Camry', year: '2020', make: 'Toyota', model: 'Camry', status: 'used', mileage: '48000' },
  { label: 'SUV · Honda CR-V', year: '2021', make: 'Honda', model: 'CR-V', status: 'certified', mileage: '31000' },
  { label: 'Truck · Ford F-150', year: '2022', make: 'Ford', model: 'F-150', status: 'used', mileage: '26000' },
  { label: 'High-mileage · Impala', year: '2008', make: 'Chevrolet', model: 'Impala', status: 'used', mileage: '189000' },
  { label: 'Luxury · BMW X5', year: '2019', make: 'BMW', model: 'X5', status: 'used', mileage: '64000' },
]
const CALIBRATION = [
  { nominal: 80, empirical: 76.32, width: '$25,885' },
  { nominal: 90, empirical: 89.1, width: '$38,435', featured: true },
  { nominal: 95, empirical: 95.64, width: '$64,028' },
]
const EXPERIMENTS = [
  ['RF05 model selection', 'Accepted', 'accepted'],
  ['Conformal calibration', 'Accepted', 'accepted'],
  ['Heteroscedastic intervals', 'Rejected', 'rejected'],
  ['Yoad moderate augmentation', 'Experimental only', 'experimental'],
  ['Yoad weighting', 'Rejected', 'rejected'],
  ['AutoTrader / KBB', 'Reference only', 'reference'],
  ['River online learning', 'Shadow validated', 'shadow'],
  ['Final holdout', 'Passed with limitations', 'limited'],
]
const RIVER_SCENARIOS = {
  stable: { label: 'Stable market', river: 1795, static: 1543, drift: null, note: 'Static RF05 remains stronger in this stable synthetic market.' },
  gradual: { label: 'Gradual shift', river: 1880, static: 6463, drift: null, note: 'The learner adapts to a smooth synthetic price drift.' },
  sudden: { label: 'Sudden shift', river: 1868, static: 8042, drift: null, note: 'Abrupt synthetic repricing separates rolling error.' },
  manufacturer: { label: 'Manufacturer shift', river: 1899, static: 3306, drift: null, note: 'A make-specific shift tests categorical adaptation.' },
  mileage: { label: 'Mileage shift', river: 3064, static: 4079, drift: 448, note: 'ADWIN emitted one telemetry event at observation 448.' },
}
const WARNING_COPY = {
  missing_mileage: 'Mileage was not supplied; the pipeline used its learned missing-value behavior.',
  rare_or_unseen_category: 'This make or model had limited support in the historical training data.',
  unsupported_feature_combination: 'This combination is outside the model’s well-supported range.',
}

function formatUsd(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', maximumFractionDigits: 0,
  }).format(value)
}

function browserClientId() {
  const key = 'autovalue-browser-id-v1'
  try {
    const existing = localStorage.getItem(key)
    if (existing) return existing
    const created = crypto.randomUUID()
    localStorage.setItem(key, created)
    return created
  } catch {
    return crypto.randomUUID()
  }
}

function ApiStatus({ status, version }) {
  return (
    <div className={`api-status api-status--${status}`} role="status" aria-live="polite">
      <span className="api-status__dot" aria-hidden="true" />
      <span>{STATUS_COPY[status]}</span>
      {version && <span className="api-status__version">v{version}</span>}
    </div>
  )
}

function Header({ view, onNavigate, api }) {
  return (
    <header className="site-header">
      <button className="brand" type="button" onClick={() => onNavigate('valuation')}>
        <span className="brand__mark" aria-hidden="true">A</span>
        <span>AutoValue <strong>AI</strong></span>
      </button>
      <nav className="site-nav" aria-label="Primary navigation">
        <button aria-current={view === 'valuation' ? 'page' : undefined} className={view === 'valuation' ? 'is-active' : ''} type="button" onClick={() => onNavigate('valuation')}>Valuation</button>
        <button aria-current={view === 'engineering' ? 'page' : undefined} className={view === 'engineering' ? 'is-active' : ''} type="button" onClick={() => onNavigate('engineering')}>ML engineering</button>
      </nav>
      <ApiStatus status={api.status} version={api.version} />
    </header>
  )
}

function VehicleForm({ form, setForm, modelReady, submitting, onSubmit }) {
  const update = (event) => setForm((current) => ({
    ...current, [event.target.name]: event.target.value,
  }))
  return (
    <form className="vehicle-form" onSubmit={onSubmit} aria-busy={submitting} aria-labelledby="vehicle-form-title">
      <div className="form-heading">
        <div><p className="overline">Vehicle information</p><h2 id="vehicle-form-title">What are you driving?</h2></div>
        <span className="step-badge">01 / 02</span>
      </div>
      <div className="example-row" aria-label="Example vehicles">
        <span>Try an example</span>
        {EXAMPLES.map((example) => (
          <button key={example.label} type="button" onClick={() => setForm((current) => ({ ...current, ...example }))}>{example.label}</button>
        ))}
      </div>
      <div className="form-grid">
        <label><span>Model year</span><input name="year" type="number" min="1900" max="2023" inputMode="numeric" value={form.year} onChange={update} required /><small>1900–2023 model years</small></label>
        <label><span>Vehicle status</span><select name="status" value={form.status} onChange={update}><option value="used">Used</option><option value="certified">Certified pre-owned</option><option value="new">New</option></select></label>
        <label><span>Make</span><input name="make" type="text" autoComplete="off" maxLength="80" value={form.make} onChange={update} placeholder="Toyota" required /></label>
        <label><span>Model</span><input name="model" type="text" autoComplete="off" maxLength="120" value={form.model} onChange={update} placeholder="Camry" required /></label>
        <label><span>Mileage</span><div className="input-unit"><input name="mileage" type="number" min="0" max="500000" inputMode="numeric" value={form.mileage} onChange={update} placeholder="48000" /><span>miles</span></div><small>Optional; up to 500,000 miles</small></label>
        <label><span>Prediction interval</span><select name="coverage" value={form.coverage} onChange={update}><option value="0.8">80% calibrated interval</option><option value="0.9">90% calibrated interval · recommended</option><option value="0.95">95% calibrated interval</option></select></label>
      </div>
      <button className="button button--primary submit-button" type="submit" disabled={!modelReady || submitting}>{submitting ? 'Estimating…' : 'Estimate vehicle'}<span aria-hidden="true">→</span></button>
      {!modelReady && <p className="form-note" role="status">Real estimates remain disabled until the checksum-verified RF05 bundle is installed.</p>}
    </form>
  )
}

function ResultPanel({ result, error, modelStatus, resultRef }) {
  if (error) return <section className="result-state result-state--error" ref={resultRef} role="alert" tabIndex="-1"><p className="overline">Unable to estimate</p><h2>{error}</h2><p>No fallback or fabricated value was returned.</p></section>
  if (!result) {
    return (
      <section className="result-state" aria-labelledby="result-state-title">
        <div className="result-orbit" aria-hidden="true"><span /></div>
        <p className="overline">Verified serving state</p>
        <h2 id="result-state-title">{modelStatus?.can_predict ? 'Ready for a vehicle' : 'Estimator artifact required'}</h2>
        <p>{modelStatus?.message || 'Checking the frozen RF05 serving bundle…'}</p>
        <div className="truth-badge"><span aria-hidden="true">✓</span> No placeholder predictions</div>
      </section>
    )
  }
  const position = Math.max(3, Math.min(97,
    ((result.predicted_value - result.interval_lower) / result.interval_width) * 100))
  return (
    <section className="valuation-result" ref={resultRef} tabIndex="-1" aria-live="polite" aria-labelledby="valuation-result-title">
      <div className="result-kicker"><span>Estimate complete</span><span>USD</span></div>
      <p className="result-label">Estimated historical asking price</p>
      <h2 id="valuation-result-title">{formatUsd(result.predicted_value)}</h2>
      <div className="range-result">
        <div><span>{Math.round(result.interval_coverage * 100)}% calibrated valuation range</span><strong>{formatUsd(result.interval_lower)} – {formatUsd(result.interval_upper)}</strong></div>
        <div className="range-result__rail" aria-hidden="true"><span style={{ left: `${position}%` }} /></div>
      </div>
      <p className="result-explanation">AutoValue estimates value from historical U.S. used-vehicle market data. The displayed range reflects calibrated model uncertainty and is not guaranteed.</p>
      {result.warnings.length > 0 && <div className="warning-list" aria-label="Data quality notes">{result.warnings.map((warning) => <p key={warning}>{WARNING_COPY[warning] || warning}</p>)}</div>}
      <dl className="result-meta"><div><dt>Model</dt><dd>RF05 · Random Forest</dd></div><div><dt>Calibration</dt><dd>Split conformal v1</dd></div><div><dt>Target</dt><dd>2023 asking price</dd></div></dl>
    </section>
  )
}

function RecentPredictions({ history }) {
  if (history.length === 0) return null
  return (
    <section className="recent-section" aria-labelledby="recent-title">
      <div><p className="overline">This browser only</p><h2 id="recent-title">Recent estimates</h2></div>
      <div className="recent-list">{history.map((item) => <article key={item.id}><span>{item.year} {item.make} {item.model}</span><strong>{formatUsd(item.predicted_value)}</strong><small>{Math.round(item.interval_coverage * 100)}% range</small></article>)}</div>
    </section>
  )
}

function AboutEstimate() {
  return (
    <section className="about-estimate" aria-labelledby="about-title">
      <div><p className="overline">Plain-language context</p><h2 id="about-title">About this estimate</h2></div>
      <div className="about-grid"><p>Estimates reflect historical advertised asking prices, not observed final transactions. Actual sale prices can differ.</p><p>Unusual, high-value, and sparsely represented vehicles carry greater uncertainty. Market conditions also change.</p><p>AutoValue is an educational ML system—not an official Kelley Blue Book, AutoTrader, appraisal, or guaranteed offer.</p></div>
    </section>
  )
}

function ValuationView({ modelStatus }) {
  const [form, setForm] = useState(INITIAL_FORM)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [history, setHistory] = useState([])
  const [clientId] = useState(browserClientId)
  const resultRef = useRef(null)
  useEffect(() => {
    const controller = new AbortController()
    fetchRecentPredictions(clientId, controller.signal).then(setHistory).catch(() => {})
    return () => controller.abort()
  }, [clientId])
  const submit = async (event) => {
    event.preventDefault(); setError(''); setSubmitting(true)
    try {
      const response = await createValuation({
        year: Number(form.year), make: form.make.trim(), model: form.model.trim(),
        vehicle_status: form.status, mileage: form.mileage === '' ? null : Number(form.mileage),
        interval_coverage: Number(form.coverage),
      }, clientId)
      setResult(response)
      setHistory(await fetchRecentPredictions(clientId))
      requestAnimationFrame(() => resultRef.current?.focus())
    } catch (requestError) {
      setResult(null); setError(requestError.message)
      requestAnimationFrame(() => resultRef.current?.focus())
    }
    finally { setSubmitting(false) }
  }
  return (
    <main id="main-content">
      <section className="product-hero">
        <div className="product-intro"><p className="eyebrow">U.S. market <span /> USD valuation</p><h1>Know the value.<span>Make the move.</span></h1><p>Model-backed estimates with calibrated uncertainty, built from historical market data—not a pricing API.</p></div>
        <div className="workspace"><VehicleForm form={form} setForm={setForm} modelReady={modelStatus?.can_predict} submitting={submitting} onSubmit={submit} /><ResultPanel result={result} error={error} modelStatus={modelStatus} resultRef={resultRef} /></div>
      </section>
      <RecentPredictions history={history} />
      <AboutEstimate />
    </main>
  )
}

function ArchitectureDiagram() {
  const lanes = [
    ['reference', 'Reference path', ['Vehicle input', 'Canonical features', 'Frozen RF05', 'Point valuation', 'Conformal calibration', 'Value + range']],
    ['shadow', 'Shadow path', ['Approved outcome', 'Governance gate', 'River learner', 'Prequential metrics', 'ADWIN telemetry']],
    ['research', 'Research path', ['External sources', 'Validation + lineage', 'Permission gate', 'Controlled experiments']],
  ]
  return <div className="architecture-grid" aria-label="AutoValue system architecture">{lanes.map(([tone, label, nodes]) => <div className={`architecture-lane architecture-lane--${tone}`} key={label}><span className="lane-label">{label}</span>{nodes.map((item, index) => <div className="architecture-node" key={item}>{item}{index < nodes.length - 1 && <i aria-hidden="true">→</i>}</div>)}</div>)}</div>
}

function CalibrationChart() {
  return <div className="calibration-chart">{CALIBRATION.map((item) => <div className={item.featured ? 'is-featured' : ''} key={item.nominal}><div className="calibration-chart__labels"><strong>{item.nominal}% nominal</strong><span>{item.empirical.toFixed(2)}% empirical</span></div><div className="calibration-chart__track"><span style={{ width: `${item.empirical}%` }} /></div><small>Mean width {item.width}</small></div>)}</div>
}

function RiverDemo() {
  const [selected, setSelected] = useState('stable')
  const [progress, setProgress] = useState(600)
  const scenario = RIVER_SCENARIOS[selected]
  useEffect(() => {
    if (progress >= 600) return undefined
    const timer = window.setInterval(() => setProgress((current) => Math.min(600, current + 15)), 55)
    return () => window.clearInterval(timer)
  }, [progress])
  const riverHeight = 92 - Math.min(70, (scenario.river / 8500) * 70)
  const staticHeight = 92 - Math.min(70, (scenario.static / 8500) * 70)
  return (
    <section className="engineering-section river-section" aria-labelledby="river-title">
      <div className="section-heading"><div><p className="overline">Shadow online learning</p><h2 id="river-title">Replay a synthetic market.</h2></div><p>Five deterministic scenarios validate test-then-train ordering, delayed outcomes, idempotent updates, restart safety, and drift telemetry. They are not live data.</p></div>
      <div className="simulation-badge">Simulation only</div>
      <div className="scenario-tabs" role="group" aria-label="River scenarios">{Object.entries(RIVER_SCENARIOS).map(([key, value]) => <button type="button" aria-pressed={selected === key} className={selected === key ? 'is-active' : ''} key={key} onClick={() => { setSelected(key); setProgress(600) }}>{value.label}</button>)}</div>
      <div className="river-console">
        <div className="river-chart"><div className="chart-grid" aria-hidden="true" /><svg viewBox="0 0 600 120" role="img" aria-label={`${scenario.label} rolling error comparison`}><path className="static-line" d={`M0,42 C150,48 280,${staticHeight + 12} 600,${staticHeight}`} /><path className="river-line" d={`M0,88 C170,78 330,${riverHeight - 10} 600,${riverHeight}`} />{scenario.drift && <line className="drift-line" x1={scenario.drift} x2={scenario.drift} y1="6" y2="108" />}</svg><div className="chart-progress" style={{ width: `${(progress / 600) * 100}%` }} /><div className="chart-legend"><span><i className="legend-static" />Static RF05</span><span><i className="legend-river" />River shadow</span></div></div>
        <div className="river-readout"><p className="overline">Verified aggregate replay</p><h3>{scenario.label}</h3><dl><div><dt>Events processed</dt><dd>{progress} / 600</dd></div><div><dt>Static rolling MAE</dt><dd>{formatUsd(scenario.static)}</dd></div><div><dt>River rolling MAE</dt><dd>{formatUsd(scenario.river)}</dd></div><div><dt>ADWIN events</dt><dd>{scenario.drift && progress >= scenario.drift ? '1 detected' : 'None yet'}</dd></div><div><dt>Model state</dt><dd>v1 · {progress} learned</dd></div></dl><p>{scenario.note}</p><button className="button button--primary" type="button" onClick={() => setProgress(0)}>Replay 600 events</button></div>
      </div>
    </section>
  )
}

function EngineeringView({ modelStatus }) {
  return (
    <main id="main-content" className="engineering-view">
      <section className="engineering-hero"><p className="eyebrow">System insights <span /> Evidence, not theater</p><h1>Inside the valuation system.</h1><p>A recruiter-facing view of the frozen model, calibrated uncertainty, governed experiments, and isolated online-learning research.</p><div className={`serving-banner ${modelStatus?.can_predict ? 'is-ready' : ''}`} role="status"><span aria-hidden="true">{modelStatus?.can_predict ? '✓' : '!'}</span><div><strong>{modelStatus?.can_predict ? 'Verified inference ready' : 'Serving boundary is fail-closed'}</strong><p>{modelStatus?.message || 'Checking local artifacts…'}</p></div></div></section>
      <section className="engineering-section" aria-labelledby="batch-title"><div className="section-heading"><div><p className="overline">Batch ML · Reference</p><h2 id="batch-title">RF05, frozen after selection.</h2></div><p>Development, calibration, and final holdout stayed isolated. The one-time holdout is permanently evaluation-only.</p></div><div className="metric-grid"><article><span>Development</span><strong>98,552</strong><small>group-safe observations</small></article><article><span>Calibration</span><strong>10,958</strong><small>separate observations</small></article><article><span>Final holdout</span><strong>27,589</strong><small>opened once</small></article><article className="metric-grid__accent"><span>Final MAE</span><strong>$10,575</strong><small>median error $6,679</small></article><article><span>Final RMSE</span><strong>$34,118</strong><small>outlier-sensitive</small></article><article><span>Final R²</span><strong>0.4176</strong><small>material limitations</small></article></div></section>
      <section className="engineering-section"><div className="section-heading"><div><p className="overline">Uncertainty calibration</p><h2>Coverage shown honestly.</h2></div><p>The default 90% split-conformal interval reached 89.10% empirical coverage. Intervals are calibrated ranges, never guarantees.</p></div><CalibrationChart /></section>
      <section className="engineering-section"><div className="section-heading"><div><p className="overline">Architecture</p><h2>Three paths. Clear boundaries.</h2></div><p>Reference inference, controlled research, and online-learning simulation cannot silently cross governance boundaries.</p></div><ArchitectureDiagram /></section>
      <section className="engineering-section"><div className="section-heading"><div><p className="overline">Experiment history</p><h2>Failed ideas stay visible.</h2></div><p>Rejected intervals and weighting treatments are first-class engineering evidence.</p></div><div className="decision-table" role="table" aria-label="Experiment decisions">{EXPERIMENTS.map(([experiment, decision, tone]) => <div role="row" key={experiment}><span role="cell">{experiment}</span><strong role="cell" className={`decision decision--${tone}`}>{decision}</strong></div>)}</div></section>
      <section className="engineering-section"><div className="section-heading"><div><p className="overline">External-data research</p><h2>More data did not mean promotion.</h2></div><p>The moderate Yoad treatment used 150,000 of 242,666 approved records. Cars MAE improved 0.87% and Yoad MAE improved 35.87%, but slice instability kept it experimental.</p></div><div className="governance-grid"><article><span className="status-dot status-dot--reference" /><h3>Cars.com-derived corpus</h3><p>Frozen RF05 reference</p></article><article><span className="status-dot status-dot--experimental" /><h3>Yoad / Craigslist</h3><p>Experimental batch only</p></article><article><span className="status-dot status-dot--reference" /><h3>Rebrowser AutoTrader</h3><p>Research / reference only</p></article><article><span className="status-dot status-dot--shadow" /><h3>River synthetic source</h3><p>Shadow simulation approved</p></article><article><span className="status-dot status-dot--blocked" /><h3>Unknown sources</h3><p>Blocked by default</p></article></div></section>
      <RiverDemo />
      <section className="engineering-section"><div className="section-heading"><div><p className="overline">Serving contract</p><h2>Five inputs. No invented importance.</h2></div><p>RF05 uses model year, make, exact model, vehicle status, and optional mileage. Global importance was not persisted, so the UI does not fabricate rankings or local explanations.</p></div><div className="feature-pills"><span>Model year</span><span>Make</span><span>Model</span><span>Status</span><span>Mileage</span><i>+ mileage/year and missingness</i></div></section>
    </main>
  )
}

function App() {
  const [view, setView] = useState(() => window.location.hash === '#engineering' ? 'engineering' : 'valuation')
  const [api, setApi] = useState({ status: 'checking', version: '' })
  const [modelStatus, setModelStatus] = useState(null)
  useEffect(() => {
    const controller = new AbortController()
    Promise.all([fetchApiHealth(controller.signal), fetchModelStatus(controller.signal)])
      .then(([health, model]) => { setApi({ status: 'online', version: health.version }); setModelStatus(model) })
      .catch((error) => { if (error.name !== 'AbortError') setApi({ status: 'offline', version: '' }) })
    return () => controller.abort()
  }, [])
  const navigate = (next) => {
    setView(next)
    window.history.replaceState(null, '', next === 'engineering' ? '#engineering' : '#valuation')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <Header view={view} onNavigate={navigate} api={api} />
      {view === 'valuation' ? <ValuationView modelStatus={modelStatus} /> : <EngineeringView modelStatus={modelStatus} />}
      <footer><span>AutoValue AI</span><span>U.S. market · USD · Open-source portfolio system · v0.1.0</span></footer>
    </div>
  )
}

export default App
