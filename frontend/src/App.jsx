import { useState, useEffect } from "react";
import { fetchHealth, fetchModelInfo, predictFlow } from "./api";
import { SAMPLE_FLOWS } from "./sampleData";

function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [inputJson, setInputJson] = useState(
    JSON.stringify(SAMPLE_FLOWS[0].features, null, 2)
  );
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recentPredictions, setRecentPredictions] = useState([]);

  // Counters derived from recent predictions
  const totalPredictions = recentPredictions.length;
  const attackCount = recentPredictions.filter((p) => p.is_anomaly).length;
  const normalCount = totalPredictions - attackCount;
  const threatRate =
    totalPredictions > 0 ? ((attackCount / totalPredictions) * 100).toFixed(1) : "0.0";

  // Check API health on mount
  useEffect(() => {
    fetchHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));

    fetchModelInfo()
      .then((info) => setModelInfo(info))
      .catch(() => {});
  }, []);

  async function handlePredict() {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const features = JSON.parse(inputJson);
      const result = await predictFlow(features);
      setPrediction(result);

      // Add to recent predictions (max 10)
      setRecentPredictions((prev) => [
        { ...result, timestamp: new Date().toLocaleTimeString() },
        ...prev.slice(0, 9),
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function loadSample(idx) {
    setInputJson(JSON.stringify(SAMPLE_FLOWS[idx].features, null, 2));
    setPrediction(null);
    setError(null);
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-header__title-group">
          <div className="app-header__icon">🛡️</div>
          <div>
            <div className="app-header__title">AI-NIDS Dashboard</div>
            <div className="app-header__subtitle">
              Network Intrusion Detection System — PyTorch
            </div>
          </div>
        </div>
        <div className="app-header__status">
          <span className={`status-dot ${apiOnline ? "" : "status-dot--offline"}`} />
          {apiOnline ? "API Online" : "API Offline"}
        </div>
      </header>

      {/* Stat Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-card__glow stat-card__glow--cyan" />
          <div className="stat-card__label">Total Predictions</div>
          <div className="stat-card__value stat-card__value--cyan">
            {totalPredictions}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-card__glow stat-card__glow--blue" />
          <div className="stat-card__label">Normal Traffic</div>
          <div className="stat-card__value stat-card__value--blue">{normalCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-card__glow stat-card__glow--red" />
          <div className="stat-card__label">Attacks Detected</div>
          <div className="stat-card__value stat-card__value--red">{attackCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-card__glow stat-card__glow--amber" />
          <div className="stat-card__label">Threat Rate</div>
          <div className="stat-card__value stat-card__value--amber">{threatRate}%</div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left: Prediction Input */}
        <div className="card">
          <div className="card__header">
            <div className="card__title">Classify Network Flow</div>
          </div>

          {/* Sample Flow Quick Buttons */}
          <div style={{ display: "flex", gap: "8px", marginBottom: "14px", flexWrap: "wrap" }}>
            {SAMPLE_FLOWS.map((s, i) => (
              <button
                key={i}
                onClick={() => loadSample(i)}
                style={{
                  background: "var(--bg-primary)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "6px",
                  padding: "6px 12px",
                  color: "var(--text-secondary)",
                  fontSize: "12px",
                  cursor: "pointer",
                  fontFamily: "var(--font-sans)",
                }}
              >
                {s.name}
              </button>
            ))}
          </div>

          <div className="predict-form">
            <textarea
              className="predict-form__textarea"
              value={inputJson}
              onChange={(e) => setInputJson(e.target.value)}
              placeholder='{"dur": 0.12, "proto": "tcp", ...}'
            />
            <button
              className="predict-form__btn"
              onClick={handlePredict}
              disabled={loading || !apiOnline}
            >
              {loading ? "Analyzing..." : "🔍 Classify Traffic"}
            </button>
          </div>

          {error && <div className="error-banner" style={{ marginTop: "14px" }}>{error}</div>}
        </div>

        {/* Right: Prediction Result */}
        <div className="card">
          <div className="card__header">
            <div className="card__title">Classification Result</div>
          </div>

          {!prediction && (
            <div style={{ color: "var(--text-muted)", fontSize: "14px", textAlign: "center", padding: "40px 0" }}>
              Submit a network flow to see classification results
            </div>
          )}

          {prediction && (
            <>
              <div className="prediction-result">
                <div className="prediction-result__label">Predicted Class</div>
                <div
                  className={`prediction-result__class ${
                    prediction.is_anomaly
                      ? "prediction-result__class--attack"
                      : "prediction-result__class--normal"
                  }`}
                >
                  {prediction.is_anomaly ? "⚠️ " : "✅ "}
                  {prediction.prediction}
                </div>
                <div className="prediction-result__confidence">
                  Confidence: {(prediction.confidence * 100).toFixed(2)}%
                </div>
              </div>

              {/* Probability Distribution Bars */}
              {prediction.class_probabilities && (
                <div className="prob-bars">
                  {Object.entries(prediction.class_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, prob], i) => (
                      <div className="prob-bar" key={cls}>
                        <div className="prob-bar__label">{cls}</div>
                        <div className="prob-bar__track">
                          <div
                            className={`prob-bar__fill ${i === 0 ? "prob-bar__fill--top" : ""}`}
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <div className="prob-bar__value">{(prob * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Recent Predictions Table */}
      <div className="card" style={{ marginBottom: "24px" }}>
        <div className="card__header">
          <div className="card__title">Recent Predictions</div>
        </div>
        {recentPredictions.length === 0 ? (
          <div style={{ color: "var(--text-muted)", fontSize: "13px", textAlign: "center", padding: "24px 0" }}>
            No predictions yet
          </div>
        ) : (
          <table className="recent-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {recentPredictions.map((p, i) => (
                <tr key={i}>
                  <td>{p.timestamp}</td>
                  <td>{p.prediction}</td>
                  <td>{(p.confidence * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`badge ${p.is_anomaly ? "badge--attack" : "badge--normal"}`}>
                      {p.is_anomaly ? "THREAT" : "SAFE"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Model Info */}
      {modelInfo && (
        <div className="card">
          <div className="card__header">
            <div className="card__title">Model Information</div>
          </div>
          <div className="model-info-grid">
            <div className="model-info-item">
              <div className="model-info-item__label">Framework</div>
              <div className="model-info-item__value">{modelInfo.framework}</div>
            </div>
            <div className="model-info-item">
              <div className="model-info-item__label">Target Classes</div>
              <div className="model-info-item__value">{modelInfo.num_classes}</div>
            </div>
            <div className="model-info-item" style={{ gridColumn: "1 / -1" }}>
              <div className="model-info-item__label">Supported Attack Categories</div>
              <div className="model-info-item__value" style={{ fontSize: "12px", lineHeight: 1.6 }}>
                {modelInfo.supported_classes.join(" · ")}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
