const API_BASE = "http://127.0.0.1:8000";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("API health check failed");
  return res.json();
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error("Failed to fetch model info");
  return res.json();
}

export async function predictFlow(features) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}
