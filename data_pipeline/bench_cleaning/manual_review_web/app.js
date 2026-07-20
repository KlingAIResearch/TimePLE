const state = { bootstrap: null, current: null };
const $ = (selector) => document.querySelector(selector);

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function renderList() {
  $("#progress").textContent = `${state.bootstrap.reviewed} / ${state.bootstrap.total} reviewed`;
  $("#samples").innerHTML = state.bootstrap.samples.map(item =>
    `<button class="sample ${item.reviewed ? "reviewed" : ""}" data-id="${item.sample_id}">${item.sample_id}: ${item.query}</button>`
  ).join("");
  document.querySelectorAll(".sample").forEach(button => button.onclick = () => loadSample(button.dataset.id));
}

function correctionToForm(correction) {
  const value = correction || {};
  $("#model-query").value = value.refined_query || "";
  $("#model-segment").value = JSON.stringify(value.refined_segment || []);
  $("#reason").value = value.reason || value.error || "";
  if (value.refined_query) $("#final-query").value = value.refined_query;
  if (Array.isArray(value.refined_segment) && value.refined_segment.length) {
    $("#final-segment").value = JSON.stringify(value.refined_segment[0] || value.refined_segment);
  }
}

async function loadSample(sampleId) {
  state.current = await request(`/api/sample?id=${encodeURIComponent(sampleId)}`);
  $("#video").src = state.current.video_url;
  $("#query").textContent = state.current.query;
  $("#gt").textContent = JSON.stringify(state.current.gt_timestamps);
  $("#final-query").value = state.current.query;
  $("#final-segment").value = JSON.stringify(state.current.gt_timestamps[0] || []);
  $("#decision").value = state.current.review?.decision || "keep";
  $("#notes").value = state.current.review?.notes || "";
  if (state.current.review?.final_query) $("#final-query").value = state.current.review.final_query;
  if (state.current.review?.final_segment?.length) $("#final-segment").value = JSON.stringify(state.current.review.final_segment[0]);
  correctionToForm(state.current.model_correction);
  $("#status").textContent = "";
}

$("#assist").onclick = async () => {
  if (!state.current) return;
  $("#status").textContent = "Running model…";
  try {
    correctionToForm(await request("/api/assist", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({sample_id: state.current.sample_id})}));
    $("#status").textContent = "Model response ready";
  } catch (error) { $("#status").textContent = error.message; }
};

$("#save").onclick = async () => {
  if (!state.current) return;
  try {
    const segment = JSON.parse($("#final-segment").value || "[]");
    await request("/api/review", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
      sample_id: state.current.sample_id,
      decision: $("#decision").value,
      final_query: $("#final-query").value,
      final_segment: segment,
      notes: $("#notes").value,
    })});
    $("#status").textContent = "Saved";
    await bootstrap();
  } catch (error) { $("#status").textContent = error.message; }
};

async function bootstrap() {
  state.bootstrap = await request("/api/bootstrap");
  renderList();
  if (!state.current && state.bootstrap.samples.length) await loadSample(state.bootstrap.samples[0].sample_id);
}

bootstrap().catch(error => $("#status").textContent = error.message);
