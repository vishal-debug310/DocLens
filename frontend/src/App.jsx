import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import "./premium-overrides.css";
import "./resume-ai-styles.css";

const API = "http://127.0.0.1:8000";
const MAX_PDF_SIZE = 20 * 1024 * 1024;

const WORKSPACES = [
  ["overview", "Overview", "⌂"],
  ["loan", "Loan Analysis", "▥"],
  ["insurance", "Insurance Analysis", "▤"],
  ["resume", "Resume & Job Matcher", "◈"],
  ["general", "General Document Analysis", "✦"],
];

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }
  if (!response.ok) {
    throw new Error(data.detail || "The backend request failed.");
  }
  return data;
}

const labelFor = category =>
  WORKSPACES.find(item => item[0] === category)?.[1] ||
  "General Document Analysis";

const blankConfig = {
  criteria: [],
  exclusions: [],
  company_name: "",
  job_title: "",
  job_description: "",
};

function validatePdf(file) {
  if (!file) return "Choose a PDF before uploading.";
  if (
    file.type !== "application/pdf" &&
    !file.name.toLowerCase().endsWith(".pdf")
  ) {
    return "Only PDF files are supported.";
  }
  if (file.size > MAX_PDF_SIZE) return "The PDF must be 20 MB or smaller.";
  if (file.size === 0) return "The selected PDF is empty.";
  return "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function App() {
  const [active, setActive] = useState("overview");
  const [documents, setDocuments] = useState([]);
  const [overview, setOverview] = useState(null);
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("doclens.selected") || "{}");
    } catch {
      return {};
    }
  });

  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [assistantQuestion, setAssistantQuestion] = useState("");
  const [assistantAnswer, setAssistantAnswer] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [configuration, setConfiguration] = useState(blankConfig);
  const [configurationStatus, setConfigurationStatus] = useState("");
  const [assistantFile, setAssistantFile] = useState(null);
  const [assistantCategory, setAssistantCategory] = useState("general");

  const fileInput = useRef(null);
  const assistantFileInput = useRef(null);

  const selectedIds = useMemo(() => selected[active] || [], [selected, active]);
  const activeLabel = labelFor(active);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const suffix =
        active === "overview" || active === "general"
          ? ""
          : `?category=${encodeURIComponent(active)}`;

      const [docs, analyses] = await Promise.all([
        request(`/api/documents${suffix}`),
        request(`/api/analyses${suffix}`),
      ]);

      setDocuments(docs.documents || []);
      setHistory(analyses.analyses || []);

      if (active === "overview") {
        setOverview(await request("/api/overview"));
      }
    } catch (err) {
      setError(err.message || "Could not load workspace data.");
    } finally {
      setLoading(false);
    }
  }, [active]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    localStorage.setItem("doclens.selected", JSON.stringify(selected));
  }, [selected]);

  useEffect(() => {
    const documentId = selectedIds[0];

    if (!documentId || !["loan", "insurance", "resume"].includes(active)) {
      setConfiguration(blankConfig);
      setJobDescription("");
      setConfigurationStatus("");
      return;
    }

    let cancelled = false;

    request(`/api/configurations/${active}/${documentId}`)
      .then(result => {
        if (cancelled) return;
        const next = { ...blankConfig, ...(result.configuration || {}) };
        setConfiguration(next);
        if (active === "resume") setJobDescription(next.job_description || "");
      })
      .catch(err => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, [active, selectedIds]);

  function navigate(category) {
    setError("");
    setAssistantAnswer(null);
    setAssistantQuestion("");
    setActive(category);
  }

  function setSelectedIds(nextIds) {
    setSelected(current => {
      const currentIds = current[active] || [];
      const resolvedIds = typeof nextIds === "function" ? nextIds(currentIds) : nextIds;
      return { ...current, [active]: resolvedIds };
    });
  }

  async function refreshCategory(category) {
    const suffix =
      category === "overview" || category === "general"
        ? ""
        : `?category=${encodeURIComponent(category)}`;

    const [docs, analyses] = await Promise.all([
      request(`/api/documents${suffix}`),
      request(`/api/analyses${suffix}`),
    ]);

    if (category === active) {
      setDocuments(docs.documents || []);
      setHistory(analyses.analyses || []);
    }

    if (category === "overview") {
      setOverview(await request("/api/overview"));
    }
  }

  async function saveConfiguration(next) {
    if (selectedIds.length !== 1) {
      setError("Select exactly one document before saving its configuration.");
      return false;
    }

    setConfigurationStatus("Saving…");
    setError("");

    try {
      const result = await request(
        `/api/configurations/${active}/${selectedIds[0]}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(next),
        }
      );

      const saved = { ...blankConfig, ...(result.configuration || next) };
      setConfiguration(saved);

      if (active === "resume") setJobDescription(saved.job_description || "");
      setConfigurationStatus("Saved");
      return true;
    } catch (err) {
      setConfigurationStatus("");
      setError(err.message);
      return false;
    }
  }

  async function removeConfiguration() {
    if (selectedIds.length !== 1) {
      setError("Select exactly one document first.");
      return;
    }

    try {
      await request(`/api/configurations/${active}/${selectedIds[0]}`, {
        method: "DELETE",
      });
      setConfiguration(blankConfig);
      if (active === "resume") setJobDescription("");
      setConfigurationStatus("Removed");
      setMessage("Saved configuration removed.");
    } catch (err) {
      setError(err.message);
    }
  }

  async function uploadFile(fileToUpload, category) {
    const validationError = validatePdf(fileToUpload);
    if (validationError) {
      setError(validationError);
      return null;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const body = new FormData();
      body.append("file", fileToUpload);
      body.append("category", category);

      const result = await request("/api/upload", {
        method: "POST",
        body,
      });

      setSelected(current => {
        const existing = current[category] || [];
        return {
          ...current,
          [category]: existing.includes(result.document_id)
            ? existing
            : [...existing, result.document_id],
        };
      });

      setMessage(
        `${result.filename} indexed with ${result.chunks_indexed} chunks in ${labelFor(category)}.`
      );

      if (category === active) await refreshCategory(category);
      if (category === "overview") await refreshCategory("overview");

      return result;
    } catch (err) {
      setError(err.message || "Upload failed.");
      return null;
    } finally {
      setLoading(false);
    }
  }

  async function upload() {
    const destination = active === "overview" ? "general" : active;
    const result = await uploadFile(file, destination);

    if (result) {
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function uploadFromAssistant() {
    const result = await uploadFile(assistantFile, assistantCategory);

    if (result) {
      setAssistantFile(null);
      if (assistantFileInput.current) assistantFileInput.current.value = "";
      setAssistantAnswer(null);
      setAssistantQuestion("");
      setActive(assistantCategory);
    }
  }

  async function deleteDocument(documentId, filename) {
    if (!window.confirm(`Delete ${filename} and all of its indexed evidence?`)) {
      return;
    }

    setError("");
    setMessage("");

    try {
      await request(`/api/documents/${documentId}`, { method: "DELETE" });

      setSelected(current => ({
        ...current,
        [active]: (current[active] || []).filter(id => id !== documentId),
      }));

      setMessage(`${filename} was deleted.`);
      await loadData();
    } catch (err) {
      setError(err.message);
    }
  }

  async function askAssistant(event) {
    event?.preventDefault();
    if (!assistantQuestion.trim()) return;

    if (active !== "overview" && selectedIds.length === 0) {
      setError("Select an existing document or upload one before asking the assistant.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const result = await request("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: assistantQuestion.trim(),
          category: active === "overview" || active === "general" ? null : active,
          document_ids: active === "overview" ? [] : selectedIds,
          job_description: active === "resume" ? jobDescription : null,
        }),
      });

      setAssistantAnswer(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function runResumeMatch(currentConfiguration = configuration) {
    if (selectedIds.length !== 1) {
      setError("Select exactly one resume.");
      return;
    }

    if (!jobDescription.trim()) {
      setError("Enter a job description before comparing the resume.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const configToSave = {
        ...currentConfiguration,
        job_description: jobDescription,
      };

      if (!(await saveConfiguration(configToSave))) return;

      const result = await request("/api/resume-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: selectedIds[0],
          job_description: jobDescription,
        }),
      });

      setAssistantAnswer({ resume: result });
      setAssistantOpen(true);
      await refreshCategory(active);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function runDocumentAnalysis(currentConfiguration) {
    if (selectedIds.length !== 1) {
      setError("Select exactly one document before running analysis.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      if (!(await saveConfiguration(currentConfiguration))) return;

      const result = await request("/api/document-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category: active,
          document_id: selectedIds[0],
          criteria: currentConfiguration.criteria,
          exclusions: currentConfiguration.exclusions,
        }),
      });

      setAssistantAnswer({ analysis: result });
      setAssistantOpen(true);
      await refreshCategory(active);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <button className="brand brand-button" onClick={() => navigate("overview")}>
          <span className="brand-mark">✦</span>
          <span>
            <strong>DocLens</strong>
            <small>DOCUMENT INTELLIGENCE</small>
          </span>
        </button>

        <p className="nav-label">WORKSPACES</p>
        <nav className="side-nav">
          {WORKSPACES.map(([key, label, icon]) => (
            <button
              key={key}
              className={`nav-item ${active === key ? "active" : ""}`}
              onClick={() => navigate(key)}
            >
              <span className="nav-icon">{icon}</span>
              <span className="nav-text">{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <div className="local-card">
          <span className="local-icon">⌘</span>
          <div>
            <strong>Local AI workspace</strong>
            <p>Your documents stay on your machine.</p>
          </div>
          <i />
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span><span>/</span><strong>{activeLabel}</strong>
          </div>
          <button className="secondary-button" onClick={loadData} disabled={loading}>
            ↻ Refresh
          </button>
        </header>

        <div className="dashboard-content">
          {error && (
            <div className="inline-message error-message" role="alert">
              <span>{error}</span>
              <button onClick={() => setError("")} aria-label="Dismiss error">×</button>
            </div>
          )}
          {message && (
            <div className="inline-message success-message" role="status">
              <span>{message}</span>
              <button onClick={() => setMessage("")} aria-label="Dismiss message">×</button>
            </div>
          )}

          {active === "overview" ? (
            <Overview data={overview} onSelect={navigate} />
          ) : (
            <Workspace
              category={active}
              documents={documents}
              history={history}
              selectedIds={selectedIds}
              setSelectedIds={setSelectedIds}
              file={file}
              setFile={setFile}
              fileInput={fileInput}
              upload={upload}
              deleteDocument={deleteDocument}
              loading={loading}
              configuration={configuration}
              configurationStatus={configurationStatus}
              saveConfiguration={saveConfiguration}
              removeConfiguration={removeConfiguration}
              runDocumentAnalysis={runDocumentAnalysis}
              jobDescription={jobDescription}
              setJobDescription={setJobDescription}
              runResumeMatch={runResumeMatch}
            />
          )}

          <footer className="app-footer">
            <span>DocLens · evidence-first local analysis</span>
            <span><i /> Local-first processing</span>
          </footer>
        </div>
      </section>

      {!assistantOpen && (
        <button
          className="assistant-fab"
          aria-label={`Open assistant for ${activeLabel}`}
          onClick={() => setAssistantOpen(true)}
        >
          <span>✦</span> AI help
        </button>
      )}

      {assistantOpen && (
        <AssistantDrawer
          activeLabel={activeLabel}
          active={active}
          selectedIds={selectedIds}
          question={assistantQuestion}
          setQuestion={setAssistantQuestion}
          answer={assistantAnswer}
          loading={loading}
          expanded={expanded}
          setExpanded={setExpanded}
          minimize={() => setAssistantOpen(false)}
          ask={askAssistant}
          assistantFile={assistantFile}
          setAssistantFile={setAssistantFile}
          assistantFileInput={assistantFileInput}
          assistantCategory={assistantCategory}
          setAssistantCategory={setAssistantCategory}
          upload={uploadFromAssistant}
        />
      )}
    </main>
  );
}

function Overview({ data, onSelect }) {
  const stats = data || {
    total_documents: 0,
    total_analyses: 0,
    analyses_today: 0,
    category_counts: {},
  };

  return (
    <section className="page-view">
      <div className="page-title-row">
        <div>
          <div className="eyebrow"><i /> PLATFORM OVERVIEW</div>
          <h1>Research, grounded in <span>evidence</span></h1>
          <p>Platform-wide counts from your indexed documents and persisted analyses.</p>
        </div>
      </div>

      <section className="stats-grid">
        {[
          ["Documents", stats.total_documents],
          ["Analyses", stats.total_analyses],
          ["Today", stats.analyses_today],
        ].map(([label, value]) => (
          <article className="stat-card" key={label}>
            <div className="stat-top">
              {label}<span className="stat-icon lavender">▤</span>
            </div>
            <strong className="stat-value">{value}</strong>
            <p>Real backend records</p>
          </article>
        ))}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <span className="heading-icon purple-icon">▤</span>
          <div>
            <h2>Documents by category</h2>
            <p>Each document is counted once using its stored category.</p>
          </div>
        </div>
        <div className="category-grid">
          {WORKSPACES.slice(1).map(([key, label]) => (
            <button className="category-card" key={key} onClick={() => onSelect(key)}>
              <strong>{label}</strong>
              <span>{stats.category_counts?.[key] || 0} documents →</span>
            </button>
          ))}
        </div>
      </section>
    </section>
  );
}

function Workspace({
  category, documents, history, selectedIds, setSelectedIds, file, setFile,
  fileInput, upload, deleteDocument, loading, configuration,
  configurationStatus, saveConfiguration, removeConfiguration,
  runDocumentAnalysis, jobDescription, setJobDescription, runResumeMatch,
}) {
  const [dragging, setDragging] = useState(false);
  const [companyName, setCompanyName] = useState(configuration.company_name || "");
  const [jobTitle, setJobTitle] = useState(configuration.job_title || "");
  const [search, setSearch] = useState("");
  const [filterCategory, setFilterCategory] = useState("");

  useEffect(() => {
    setCompanyName(configuration.company_name || "");
    setJobTitle(configuration.job_title || "");
  }, [configuration.company_name, configuration.job_title]);

  const toggle = id => setSelectedIds(currentIds => {
    if (category === "resume") {
      // Resume comparison requires exactly one resume; selecting another replaces it.
      return currentIds.includes(id) ? [] : [id];
    }
    return currentIds.includes(id)
      ? currentIds.filter(item => item !== id)
      : [...currentIds, id];
  });

  const visibleDocuments = documents
    .filter(doc => !search.trim() ||
      doc.filename.toLowerCase().includes(search.trim().toLowerCase()))
    .filter(doc => !filterCategory || doc.category === filterCategory);

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0] || null;
    if (selectedFile) {
      const validationError = validatePdf(selectedFile);
      if (validationError) {
        setFile(null);
        event.target.value = "";
        return;
      }
    }
    setFile(selectedFile);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    const droppedFile = event.dataTransfer.files?.[0] || null;
    const validationError = droppedFile ? validatePdf(droppedFile) : "";
    if (validationError) {
      setFile(null);
      return;
    }
    setFile(droppedFile);
  }

  return (
    <section className="page-view">
      <div className="page-title-row">
        <div>
          <div className="eyebrow"><i /> {labelFor(category).toUpperCase()}</div>
          <h1>{labelFor(category)} <span>workspace</span></h1>
          <p>Choose the evidence this workspace may use before starting analysis.</p>
        </div>
      </div>

      <section className="panel">
        <div className="panel-heading">
          <span className="heading-icon purple-icon">↥</span>
          <div>
            <h2>Upload a document</h2>
            <p>PDF files are processed locally and assigned to this workspace.</p>
          </div>
          <span className="step-pill">LOCAL INDEX</span>
        </div>

        <label
          className={`dropzone ${dragging ? "dropzone-active" : ""}`}
          htmlFor="document-upload"
          onDragOver={event => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <span className="upload-illustration">↑</span>
          <strong>{file ? file.name : "Drop your PDF here"}</strong>
          <span className="drop-hint">
            {file ? `${formatBytes(file.size)} · Ready to upload` : "or click to browse your computer"}
          </span>
          <span className="file-types"><b>PDF</b> Text-based PDF · Max 20 MB</span>
          <input
            ref={fileInput}
            id="document-upload"
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
          />
        </label>

        <div className="upload-actions">
          <span className="privacy-note">▣ Processed by your local AI setup</span>
          <button className="primary-button" onClick={upload} disabled={loading || !file}>
            {loading ? "Indexing document…" : "Upload and index"} <span>→</span>
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <span className="heading-icon green-icon">▤</span>
          <div>
            <h2>Select evidence</h2>
            <p>{selectedIds.length
              ? `${selectedIds.length} selected for this session`
              : "Select existing documents before asking the assistant."}</p>
          </div>
          <span className="step-pill">{documents.length} FOUND</span>
        </div>

        {category === "general" && (
          <div className="document-filters">
            <input
              value={search}
              onChange={event => setSearch(event.target.value)}
              placeholder="Search by filename…"
              aria-label="Search documents by filename"
            />
            <select
              value={filterCategory}
              onChange={event => setFilterCategory(event.target.value)}
              aria-label="Filter documents by category"
            >
              <option value="">All categories</option>
              <option value="loan">Loan</option>
              <option value="insurance">Insurance</option>
              <option value="resume">Resume</option>
              <option value="general">General</option>
            </select>
          </div>
        )}

        {documents.length === 0 ? (
          <div className="page-empty">
            <strong>No documents in this workspace</strong>
            <p>Upload a document above to begin.</p>
          </div>
        ) : visibleDocuments.length === 0 ? (
          <div className="page-empty">
            <strong>No matching documents</strong>
            <p>Try another filename or category filter.</p>
          </div>
        ) : (
          <div className="documents-list">
            {visibleDocuments.map(doc => (
              <article
                className={`document-row selectable-document ${selectedIds.includes(doc.document_id) ? "selected-document" : ""}`}
                key={doc.document_id}
              >
                <input
                  type="checkbox"
                  checked={selectedIds.includes(doc.document_id)}
                  onChange={() => toggle(doc.document_id)}
                  aria-label={`Select ${doc.filename}`}
                />
                <span className="document-file-icon">PDF</span>
                <div className="document-info" onClick={() => toggle(doc.document_id)}>
                  <strong>{doc.filename}</strong>
                  <p>{doc.category} · {doc.chunks_indexed} chunks · {doc.pages?.length || 0} pages · {doc.processing_status || "completed"}</p>
                </div>
                <button className="text-button danger" onClick={() => deleteDocument(doc.document_id, doc.filename)}>
                  Delete
                </button>
              </article>
            ))}
          </div>
        )}
      </section>

      {["loan", "insurance"].includes(category) && (
        <ConfigurationEditor
          category={category}
          configuration={configuration}
          status={configurationStatus}
          onSave={saveConfiguration}
          onRemove={removeConfiguration}
          onRun={runDocumentAnalysis}
          loading={loading}
        />
      )}

      {category === "resume" && (
        <section className="panel resume-panel">
          <div className="panel-heading">
            <span className="heading-icon blue-icon">◈</span>
            <div>
              <h2>Resume-to-job comparison</h2>
              <p>Enter the target role and job description, then compare against one selected resume.</p>
            </div>
          </div>

          <div className="resume-form-grid">
            <label className="field-group">
              <span>Company name <small>(optional)</small></span>
              <input value={companyName} onChange={event => setCompanyName(event.target.value)} placeholder="e.g. Virtusa" />
            </label>
            <label className="field-group">
              <span>Job title <small>(optional)</small></span>
              <input value={jobTitle} onChange={event => setJobTitle(event.target.value)} placeholder="e.g. Associate Software Engineer" />
            </label>
          </div>

          <label className="field-group job-description-field">
            <span>Job description <b>*</b></span>
            <textarea
              value={jobDescription}
              onChange={event => setJobDescription(event.target.value)}
              placeholder="Paste the job description here. Include responsibilities, required skills and qualifications…"
              rows={7}
            />
          </label>

          <div className="form-actions">
            <button
              className="secondary-button"
              onClick={() => saveConfiguration({
                ...configuration,
                company_name: companyName.trim(),
                job_title: jobTitle.trim(),
                job_description: jobDescription,
              })}
              disabled={loading || selectedIds.length !== 1}
            >
              Save job details
            </button>
            <button className="text-button danger" onClick={removeConfiguration} disabled={loading || selectedIds.length !== 1}>
              Remove saved job
            </button>
            <button
              className="primary-button"
              onClick={() => runResumeMatch({
                ...configuration,
                company_name: companyName.trim(),
                job_title: jobTitle.trim(),
                job_description: jobDescription,
              })}
              disabled={loading || selectedIds.length !== 1 || !jobDescription.trim()}
              title={loading ? "Please wait for the current operation" : selectedIds.length !== 1 ? "Select one resume above" : !jobDescription.trim() ? "Enter a job description first" : "Compare the selected resume with this job description"}
            >
              {loading ? "Comparing…" : "Compare selected resume →"}
            </button>
          </div>

          {configurationStatus && <small className="saved-state">{configurationStatus}</small>}
          {selectedIds.length !== 1 && (
            <p className="form-help">Select exactly one resume in the Select evidence section to save job details or compare.</p>
          )}
        </section>
      )}

      {history.length > 0 && (
        <section className="panel">
          <div className="panel-heading">
            <span className="heading-icon amber">◷</span>
            <div>
              <h2>Analysis history</h2>
              <p>Persisted results for this category.</p>
            </div>
          </div>
          {history.map(item => (
            <div className="history-row" key={item.analysis_id}>
              <strong>{item.filename}</strong>
              <span>{item.match_percentage != null ? `${item.match_percentage}% estimated match` : item.status}</span>
              <small>{new Date(item.created_at).toLocaleString()}</small>
            </div>
          ))}
        </section>
      )}
    </section>
  );
}

function ConfigurationEditor({ category, configuration, status, onSave, onRemove, onRun, loading }) {
  const [draft, setDraft] = useState(configuration);
  useEffect(() => setDraft(configuration), [configuration]);

  function updateCriterion(id, field, value) {
    setDraft(current => ({
      ...current,
      criteria: current.criteria.map(item => item.id === id ? { ...item, [field]: value } : item),
    }));
  }

  function addCriterion() {
    setDraft(current => ({
      ...current,
      criteria: [...current.criteria, { id: crypto.randomUUID(), name: "", instructions: "" }],
    }));
  }

  function addExclusion() {
    setDraft(current => ({
      ...current,
      exclusions: [...current.exclusions, { id: crypto.randomUUID(), value: "" }],
    }));
  }

  return (
    <section className="panel configuration-panel">
      <div className="panel-heading">
        <span className="heading-icon blue-icon">☷</span>
        <div>
          <h2>{category === "loan" ? "Loan" : "Insurance"} analysis configuration</h2>
          <p>User-defined instructions guide analysis; they do not prove approval or claim eligibility.</p>
        </div>
      </div>

      <div className="config-section">
        <div className="config-heading">
          <strong>Criteria</strong>
          <button className="secondary-button" onClick={addCriterion}>+ Add criterion</button>
        </div>
        {draft.criteria.length === 0 && <p className="empty-config">No criteria yet. Add one to define what the analysis should inspect.</p>}
        {draft.criteria.map(item => (
          <div className="config-item" key={item.id}>
            <input value={item.name} onChange={event => updateCriterion(item.id, "name", event.target.value)} placeholder="Criterion name" />
            <textarea value={item.instructions} onChange={event => updateCriterion(item.id, "instructions", event.target.value)} placeholder="What should be checked and what evidence is relevant?" />
            <button className="text-button danger" onClick={() => setDraft(current => ({ ...current, criteria: current.criteria.filter(candidate => candidate.id !== item.id) }))}>Remove</button>
          </div>
        ))}
      </div>

      <div className="config-section">
        <div className="config-heading">
          <strong>Ignore / Exclude</strong>
          <button className="secondary-button" onClick={addExclusion}>+ Add exclusion</button>
        </div>
        {draft.exclusions.map(item => (
          <div className="config-item exclusion-item" key={item.id}>
            <input
              value={item.value}
              onChange={event => setDraft(current => ({
                ...current,
                exclusions: current.exclusions.map(candidate => candidate.id === item.id ? { ...candidate, value: event.target.value } : candidate),
              }))}
              placeholder="Term or topic to exclude"
            />
            <button className="text-button danger" onClick={() => setDraft(current => ({ ...current, exclusions: current.exclusions.filter(candidate => candidate.id !== item.id) }))}>Remove</button>
          </div>
        ))}
      </div>

      <div className="form-actions">
        <button className="secondary-button" onClick={() => onSave(draft)} disabled={loading}>Save configuration</button>
        <button className="primary-button" onClick={() => onRun(draft)} disabled={loading}>Run analysis</button>
        <button className="text-button danger" onClick={onRemove} disabled={loading}>Remove saved configuration</button>
        {status && <small className="saved-state">{status}</small>}
      </div>
    </section>
  );
}

function AssistantDrawer({
  activeLabel, active, selectedIds, question, setQuestion, answer, loading,
  expanded, setExpanded, minimize, ask, assistantFile, setAssistantFile,
  assistantFileInput, assistantCategory, setAssistantCategory, upload,
}) {
  const context = active === "overview"
    ? "Platform overview · metrics and recent activity"
    : `${activeLabel} · ${selectedIds.length} selected document${selectedIds.length === 1 ? "" : "s"}`;

  const suggestions = active === "overview"
    ? ["How many documents are indexed?", "What analyses were completed today?", "Show the current workspace status"]
    : active === "loan"
      ? ["What income evidence is present?", "Which loan obligations are mentioned?", "What information is missing?"]
      : active === "insurance"
        ? ["What coverage is described?", "What exclusions are listed?", "What claim requirements are mentioned?"]
        : active === "resume"
          ? ["Which job requirements match the resume?", "What skills are weakly evidenced?", "What improvements are suggested?"]
          : ["Summarize the selected document", "What are the key findings?", "What information is missing?"];

  function chooseAssistantFile(event) {
    const selectedFile = event.target.files?.[0] || null;
    if (!selectedFile) {
      setAssistantFile(null);
      return;
    }

    const validationError = validatePdf(selectedFile);
    if (validationError) {
      setAssistantFile(null);
      event.target.value = "";
      return;
    }
    setAssistantFile(selectedFile);
  }

  return (
    <section className={`assistant-drawer ${expanded ? "expanded" : ""}`} role="dialog" aria-label="AI research assistant">
      <header className="assistant-header">
        <span className="assistant-avatar">✦</span>
        <div className="assistant-title">
          <strong>Research assistant</strong>
          <small><i /> {context}</small>
        </div>
        <div className="assistant-controls">
          <button aria-label={expanded ? "Shrink assistant" : "Expand assistant"} onClick={() => setExpanded(value => !value)}>
            {expanded ? "⊡" : "□"}
          </button>
          <button aria-label="Minimize assistant" title="Minimize" onClick={minimize}>−</button>
        </div>
      </header>

      <div className="assistant-body">
        <div className="assistant-intro">
          <span>✦</span>
          <small>PAGE-AWARE LOCAL AI</small>
          <h2>Ask with confidence</h2>
          <p>{active === "overview"
            ? "Ask about platform metrics and recent activity."
            : selectedIds.length
              ? "Answers are restricted to your selected evidence."
              : "Select a document below before asking document questions."}</p>
        </div>

        <div className="assistant-upload">
          <div className="assistant-upload-heading">
            <strong>Upload a PDF</strong>
            <small>Choose where the document should be stored.</small>
          </div>

          <label className="assistant-upload-select">
            <span>Destination workspace</span>
            <select value={assistantCategory} onChange={event => setAssistantCategory(event.target.value)}>
              <option value="loan">Loan</option>
              <option value="insurance">Insurance</option>
              <option value="resume">Resume</option>
              <option value="general">General</option>
            </select>
          </label>

          <input
            ref={assistantFileInput}
            className="assistant-file-input"
            id="assistant-pdf-upload"
            type="file"
            accept=".pdf,application/pdf"
            onChange={chooseAssistantFile}
          />

          <label className="assistant-file-picker" htmlFor="assistant-pdf-upload">
            <span className="assistant-file-picker-icon">↑</span>
            <span className="assistant-file-picker-text">
              <strong>{assistantFile ? assistantFile.name : "Choose a PDF file"}</strong>
              <small>{assistantFile ? `${formatBytes(assistantFile.size)} · Ready to upload` : "PDF only · Maximum 20 MB"}</small>
            </span>
            <span className="assistant-browse">Browse</span>
          </label>

          <button type="button" className="primary-button assistant-upload-button" onClick={upload} disabled={loading || !assistantFile}>
            {loading ? "Uploading…" : "Upload to workspace"} <span>→</span>
          </button>
        </div>

        <div className="suggestions">
          <span className="suggestion-label">SUGGESTED QUESTIONS</span>
          {suggestions.map(item => (
            <button type="button" key={item} onClick={() => setQuestion(item)}>
              {item}<span>→</span>
            </button>
          ))}
        </div>

        {answer?.resume && (
          <div className="assistant-answer resume-ai-result">
            <div className="resume-score-header">
              <div>
                <span className="resume-eyebrow">SEMANTIC RESUME ANALYSIS</span>
                <strong className="resume-score">{answer.resume.match_percentage}% estimated match</strong>
              </div>
              <span className="resume-score-caption">Evidence-weighted estimate</span>
            </div>
            {answer.resume.summary && <p className="resume-summary">{answer.resume.summary}</p>}
            <p className="resume-method-note">{answer.resume.method}</p>

            <div className="resume-skill-grid">
              <section className="resume-result-section">
                <h4>Strong matches <span>{answer.resume.matched_skills?.length || 0}</span></h4>
                {answer.resume.matched_skills?.length ? <ul>{answer.resume.matched_skills.map((skill, index) => <li key={`${skill}-${index}`}>{skill}</li>)}</ul> : <p>No strong matches identified.</p>}
              </section>
              <section className="resume-result-section">
                <h4>Partial matches <span>{answer.resume.partial_skills?.length || 0}</span></h4>
                {answer.resume.partial_skills?.length ? <ul>{answer.resume.partial_skills.map((skill, index) => <li key={`${skill}-${index}`}>{skill}</li>)}</ul> : <p>No partial matches identified.</p>}
              </section>
              <section className="resume-result-section">
                <h4>Missing requirements <span>{answer.resume.missing_skills?.length || 0}</span></h4>
                {answer.resume.missing_skills?.length ? <ul>{answer.resume.missing_skills.map((skill, index) => <li key={`${skill}-${index}`}>{skill}</li>)}</ul> : <p>No missing requirements identified.</p>}
              </section>
            </div>

            {!!answer.resume.requirements?.length && (
              <section className="resume-result-section resume-requirements">
                <h4>Requirement-by-requirement evidence</h4>
                {answer.resume.requirements.map((item, index) => (
                  <article className="resume-requirement" key={`${item.skill}-${index}`}>
                    <div className="resume-requirement-top">
                      <strong>{item.skill}</strong>
                      <span className={`resume-status status-${item.status}`}>{item.status}</span>
                    </div>
                    <p><b>Priority:</b> {item.importance}</p>
                    {item.resume_evidence && <p><b>Resume evidence:</b> {item.resume_evidence}</p>}
                    {item.reason && <p><b>Assessment:</b> {item.reason}</p>}
                  </article>
                ))}
              </section>
            )}

            {!!answer.resume.strengths?.length && (
              <section className="resume-result-section">
                <h4>Resume strengths</h4>
                <ul>{answer.resume.strengths.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
              </section>
            )}
            {!!answer.resume.recommendations?.length && (
              <section className="resume-result-section">
                <h4>Recommended improvements</h4>
                <ul>{answer.resume.recommendations.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>
              </section>
            )}
          </div>
        )}

        {answer?.analysis && (
          <div className="assistant-answer">
            <strong>Analysis result</strong>
            <p>{answer.analysis.summary || answer.analysis.answer || "Analysis completed. Check analysis history for details."}</p>
          </div>
        )}

        {answer?.answer && (
          <div className="assistant-answer">
            <strong>Answer</strong>
            <p>{answer.answer}</p>
            {answer.sources?.length > 0 && (
              <div className="assistant-sources">
                <strong>Sources</strong>
                {answer.sources.map(source => (
                  <div key={`${source.filename}-${source.page}`}>
                    <b>PDF</b>
                    <span>{source.filename}<small>Page {source.page}</small></span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <form className="assistant-composer" onSubmit={ask}>
        <textarea
          value={question}
          onChange={event => setQuestion(event.target.value)}
          placeholder={active === "overview"
            ? "Ask about your workspace…"
            : selectedIds.length
              ? "Ask about selected documents…"
              : "Select a document to begin…"}
        />
        <div>
          <small><i /> {loading ? "Working…" : "Evidence-filtered context"}</small>
          <button type="submit" disabled={loading || !question.trim()} aria-label="Send question">→</button>
        </div>
      </form>
    </section>
  );
}
