import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ApiClient, ApiError, saveBlob } from "./api/client";
import {
  AnswerPanel,
  AssessmentContent,
  EmptyState,
  Icon,
  Loading,
  Notice,
  PageHeader,
} from "./components";

const TOKEN_KEY = "genai_platform_token";
const MAX_MOCK_CHAPTERS = 8;
const MAX_MOCK_TOPICS = 8;

const teacherNavigation = [
  ["home", "Overview", "home"],
  ["generate", "Assessment generator", "assessment"],
  ["mock-test", "25-mark mock test", "mock"],
  ["assessments", "My assessments", "library"],
  ["grade-review", "Grade review", "grades"],
  ["rag", "RAG assistant", "chat"],
  ["monitoring", "RAG monitoring", "monitoring"],
];

const studentNavigation = [
  ["home", "My learning", "home"],
  ["rag", "Student RAG Assistant", "chat"],
  ["assessments", "Assessments", "assessment"],
  ["grades", "My grades", "grades"],
];

function useAsyncData(loader, dependencies = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  useEffect(() => {
    let current = true;
    setState((previous) => ({ ...previous, loading: true, error: null }));
    loader()
      .then((data) => current && setState({ data, loading: false, error: null }))
      .catch((error) => current && setState({ data: null, loading: false, error }));
    return () => { current = false; };
    // The caller controls when a resource should be reloaded.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);
  return state;
}

function AuthScreen({ api, onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ email: "", password: "", role: "student" });
  const [status, setStatus] = useState({ loading: false, message: "", type: "" });

  const submit = async (event) => {
    event.preventDefault();
    setStatus({ loading: true, message: "", type: "" });
    try {
      if (mode === "register") {
        await api.post("/auth/register", form);
        setMode("login");
        setStatus({ loading: false, message: "Account created. You can now sign in.", type: "success" });
        return;
      }
      const result = await api.post("/auth/login", { email: form.email, password: form.password });
      localStorage.setItem(TOKEN_KEY, result.access_token);
      const user = await api.get("/auth/me");
      onAuthenticated(user);
    } catch (error) {
      setStatus({ loading: false, message: error.message, type: "error" });
    }
  };

  return (
    <main className="auth-layout">
      <section className="auth-story">
        <div className="brand brand-large"><span className="brand-mark">CS</span><span>GenAI Learn</span></div>
        <div className="auth-copy">
          <p className="eyebrow">Cambridge Computer Science · O Level & A Level</p>
          <h1>Learn deeply.<br />Assess intelligently.</h1>
          <p>
            A source-grounded learning platform combining curriculum-aware RAG,
            teacher-reviewed assessments, and transparent grading.
          </p>
        </div>
        <div className="auth-feature-row">
          <span>Grounded answers</span><span>Teacher oversight</span><span>Measurable quality</span>
        </div>
      </section>
      <section className="auth-panel">
        <div className="auth-card">
          <p className="eyebrow">{mode === "login" ? "Welcome back" : "Join the platform"}</p>
          <h2>{mode === "login" ? "Sign in to continue" : "Create your account"}</h2>
          <p className="muted">
            {mode === "login" ? "Access your personalized learning workspace." : "Choose the role that matches how you will use the platform."}
          </p>
          <div className="auth-tabs" role="tablist">
            <button className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>Sign in</button>
            <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>Create account</button>
          </div>
          <form onSubmit={submit} className="stack-form">
            {mode === "register" && (
              <label>
                I am a
                <div className="role-picker">
                  {["student", "teacher"].map((role) => (
                    <button
                      type="button"
                      className={form.role === role ? "selected" : ""}
                      key={role}
                      onClick={() => setForm({ ...form, role })}
                    >
                      <Icon name={role === "student" ? "user" : "assessment"} />
                      <span><strong>{role[0].toUpperCase() + role.slice(1)}</strong><small>{role === "student" ? "Learn and complete work" : "Create and review work"}</small></span>
                    </button>
                  ))}
                </div>
              </label>
            )}
            <label>Email address<input type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} placeholder="you@example.com" /></label>
            <label>Password<input type="password" required minLength={mode === "register" ? 8 : 1} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} placeholder={mode === "register" ? "At least 8 characters" : "Your password"} /></label>
            {status.message && <Notice type={status.type}>{status.message}</Notice>}
            <button className="primary-button full-width" disabled={status.loading}>
              {status.loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}

function Sidebar({ user, active, onNavigate, onLogout, open, onClose }) {
  const links = user.role === "teacher" ? teacherNavigation : studentNavigation;
  return (
    <>
      {open && <button className="sidebar-scrim" aria-label="Close navigation" onClick={onClose} />}
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="brand"><span className="brand-mark">CS</span><span>GenAI Learn</span></div>
        <nav aria-label="Main navigation">
          <p className="nav-label">Workspace</p>
          {links.map(([id, label, icon]) => (
            <button key={id} className={active === id ? "active" : ""} onClick={() => { onNavigate(id); onClose(); }}>
              <Icon name={icon} /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-user">
          <div className="avatar">{user.email[0].toUpperCase()}</div>
          <div><strong>{user.role === "teacher" ? "Teacher account" : "Student account"}</strong><small>{user.email}</small></div>
          <button className="icon-button" onClick={onLogout} title="Sign out" aria-label="Sign out"><Icon name="logout" /></button>
        </div>
      </aside>
    </>
  );
}

function Topbar({ user, onMenu }) {
  return (
    <div className="topbar">
      <button className="mobile-menu icon-button" onClick={onMenu} aria-label="Open navigation"><Icon name="menu" /></button>
      <p>{user.role === "teacher" ? "Teacher workspace" : "Student learning hub"}</p>
      <span className="role-badge">{user.role}</span>
    </div>
  );
}

function StatCard({ label, value, note, tone = "" }) {
  return (
    <article className={`stat-card ${tone}`}>
      <p>{label}</p><strong>{value ?? "—"}</strong>{note && <small>{note}</small>}
    </article>
  );
}

function TeacherHome({ api, navigate }) {
  const { data, loading, error } = useAsyncData(() => api.get("/dashboard/teacher"), []);
  const summary = data?.summary || {};
  return (
    <>
      <PageHeader eyebrow="Good to see you" title="Teacher overview" description="Generate grounded learning material, publish it deliberately, and keep final grading decisions with the teacher." />
      {error && <Notice type="error">{error.message}</Notice>}
      <div className="stats-grid">
        <StatCard label="Students" value={loading ? "…" : summary.students} note="Registered learners" />
        <StatCard label="Drafts to review" value={loading ? "…" : summary.pending_assessments} note="Approval required" tone="accent" />
        <StatCard label="Grades to review" value={loading ? "…" : summary.pending_grade_reviews} note="Human decision pending" />
        <StatCard label="Published learning" value={loading ? "…" : summary.approved_assessments ?? "—"} note="Available to students" />
      </div>
      <section className="section-block">
        <div className="section-heading"><div><p className="eyebrow">Create</p><h2>What would you like to build?</h2></div></div>
        <div className="action-grid">
          <button className="action-card" onClick={() => navigate("generate")}><span className="action-icon"><Icon name="assessment" /></span><h3>Quick assessment</h3><p>Create a quiz or assignment from a focused curriculum topic.</p><span className="text-link">Start generating →</span></button>
          <button className="action-card" onClick={() => navigate("mock-test")}><span className="action-icon"><Icon name="mock" /></span><h3>25-mark mock test</h3><p>Select syllabus chapters and topics for a structured practice test.</p><span className="text-link">Build mock test →</span></button>
          <button className="action-card" onClick={() => navigate("rag")}><span className="action-icon"><Icon name="chat" /></span><h3>Ask the RAG assistant</h3><p>Explore the ingested books, papers, mark schemes, and syllabus.</p><span className="text-link">Ask a question →</span></button>
        </div>
      </section>
    </>
  );
}

function StudentHome({ api, navigate }) {
  const { data, error } = useAsyncData(() => api.get("/dashboard/student"), []);
  const summary = data?.summary || {};
  return (
    <>
      <PageHeader eyebrow="Your learning space" title="Ready to learn?" description="Ask grounded questions, complete teacher-published work, and track reviewed results." />
      {error && <Notice type="error">{error.message}</Notice>}
      <section className="student-hero">
        <div>
          <span className="hero-symbol">✦</span>
          <p className="eyebrow">Student RAG Assistant</p>
          <h2>Turn a difficult topic into a clear explanation.</h2>
          <p>Ask questions based on the course books and supporting Cambridge material, with sources kept available for inspection.</p>
          <button className="primary-button" onClick={() => navigate("rag")}>Start asking questions</button>
        </div>
        <div className="suggestion-card">
          <p>Try asking</p>
          {["Explain binary search step by step.", "Compare a compiler and an interpreter.", "Show me how SQL SELECT works."].map((question) => (
            <button key={question} onClick={() => navigate("rag", question)}>{question}<span>→</span></button>
          ))}
        </div>
      </section>
      <div className="stats-grid compact">
        <StatCard label="Available assessments" value={summary.approved_assessments ?? "—"} note="Published by teachers" />
        <StatCard label="Learning sources" value="25" note="Books and Cambridge material" />
        <StatCard label="Support" value="24/7" note="Curriculum-grounded assistant" />
      </div>
    </>
  );
}

function RagAssistant({ api, role, initialQuestion = "" }) {
  const [question, setQuestion] = useState(initialQuestion);
  const [difficulty, setDifficulty] = useState("Intermediate");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const ask = async (event) => {
    event.preventDefault();
    if (!question.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      setResult(await api.post("/chat", { query: question.trim(), difficulty }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageHeader eyebrow={role === "student" ? "Learn with evidence" : "Explore the corpus"} title={role === "student" ? "Student RAG Assistant" : "RAG Assistant"} description="Ask a Computer Science question and receive a response grounded in the indexed learning material." />
      <section className="chat-workspace">
        <form className="ask-box" onSubmit={ask}>
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about a concept, algorithm, past-paper question, or pseudocode…" rows="5" />
          <div className="ask-actions">
            <label>Explanation depth<select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}><option>Beginner</option><option>Intermediate</option><option>Advanced</option></select></label>
            <button className="primary-button" disabled={loading || !question.trim()}>{loading ? "Thinking…" : "Ask assistant"} <span>✦</span></button>
          </div>
        </form>
        {error && <Notice type="error">{error}</Notice>}
        {loading && <Loading label="Retrieving evidence and preparing your answer…" />}
        <AnswerPanel result={result} />
        {!result && !loading && !error && (
          <div className="prompt-suggestions">
            <p>Suggested questions</p>
            <div>{["How does binary search work?", "Explain validation vs verification.", "Write an SQL query with a WHERE condition."].map((item) => <button key={item} onClick={() => setQuestion(item)}>{item}</button>)}</div>
          </div>
        )}
      </section>
    </>
  );
}

function AssessmentGenerator({ api, onCreated }) {
  const [form, setForm] = useState({ topic: "", difficulty: "medium", assessment_type: "quiz", question_count: 5, question_format: "mixed", level: "O_LEVEL" });
  const [state, setState] = useState({ loading: false, result: null, error: null });
  const submit = async (event) => {
    event.preventDefault();
    setState({ loading: true, result: null, error: null });
    try {
      const result = await api.post("/assessments", form);
      setState({ loading: false, result, error: null });
      onCreated?.();
    } catch (error) {
      setState({ loading: false, result: null, error });
    }
  };
  return (
    <>
      <PageHeader eyebrow="Teacher-reviewed generation" title="Assessment generator" description="Create a grounded quiz or assignment draft. Nothing reaches students until you approve it." />
      <div className="two-column-layout">
        <form className="panel stack-form" onSubmit={submit}>
          <div className="panel-heading"><h2>Assessment blueprint</h2><p>Define the learning target and expected question style.</p></div>
          <label className="field-wide">Topic<input required minLength="2" value={form.topic} onChange={(e) => setForm({ ...form, topic: e.target.value })} placeholder="e.g. binary search" /></label>
          <div className="form-grid">
            <label>Level<select value={form.level} onChange={(e) => setForm({ ...form, level: e.target.value })}><option value="O_LEVEL">O Level</option><option value="A_LEVEL">A Level</option></select></label>
            <label>Difficulty<select value={form.difficulty} onChange={(e) => setForm({ ...form, difficulty: e.target.value })}><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select></label>
            <label>Assessment type<select value={form.assessment_type} onChange={(e) => setForm({ ...form, assessment_type: e.target.value })}><option value="quiz">Quiz</option><option value="assignment">Assignment</option></select></label>
            <label>Question format<select value={form.question_format} onChange={(e) => setForm({ ...form, question_format: e.target.value })}><option value="mixed">Mixed</option><option value="mcq">Multiple choice</option><option value="short_answer">Short answer</option><option value="long_answer">Long answer</option></select></label>
            <label className="field-wide">Number of questions <span>{form.question_count}</span><input type="range" min="1" max="12" value={form.question_count} onChange={(e) => setForm({ ...form, question_count: Number(e.target.value) })} /></label>
          </div>
          <button className="primary-button" disabled={state.loading}>{state.loading ? "Generating…" : "Generate teacher-review draft"}</button>
          {state.error && <ValidationError error={state.error} />}
        </form>
        <section>
          {state.loading && <Loading label="Retrieving curriculum evidence and generating the draft…" />}
          {state.result ? <AssessmentContent content={state.result.content} includeAnswers /> : !state.loading && <EmptyState title="Your draft will appear here" description="The generated questions, model answers, rubrics, and evidence will be ready for review." />}
        </section>
      </div>
    </>
  );
}

function ValidationError({ error }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div>
      <Notice type="error">{error.message}</Notice>
      {error.details && (
        <button type="button" className="text-button" onClick={() => setExpanded(!expanded)}>{expanded ? "Hide" : "View"} validation diagnostics</button>
      )}
      {expanded && <pre className="diagnostics">{JSON.stringify({ validation: error.details.validation, planner: error.details.blueprint }, null, 2)}</pre>}
    </div>
  );
}

function MockTestGenerator({ api, onCreated }) {
  const [level, setLevel] = useState("O_LEVEL");
  const [chapters, setChapters] = useState([]);
  const [chapterIds, setChapterIds] = useState([]);
  const [topicIds, setTopicIds] = useState([]);
  const [difficulty, setDifficulty] = useState("medium");
  const [state, setState] = useState({ loading: false, result: null, error: null });
  const syllabus = useAsyncData(() => api.get(`/syllabus/${level}/chapters`), [level]);

  useEffect(() => {
    setChapters(syllabus.data?.chapters || []);
    setChapterIds([]);
    setTopicIds([]);
  }, [syllabus.data]);

  const selectedChapters = chapters.filter((chapter) => chapterIds.includes(chapter.id));
  const availableTopics = selectedChapters.flatMap((chapter) => chapter.topics);
  const toggleChapter = (chapterId) => {
    if (!chapterIds.includes(chapterId) && chapterIds.length >= MAX_MOCK_CHAPTERS) {
      setState({ loading: false, result: null, error: new ApiError(`Select no more than ${MAX_MOCK_CHAPTERS} chapters per mock test.`) });
      return;
    }
    const nextChapterIds = chapterIds.includes(chapterId)
      ? chapterIds.filter((item) => item !== chapterId)
      : [...chapterIds, chapterId];
    const validTopicIds = new Set(
      chapters
        .filter((chapter) => nextChapterIds.includes(chapter.id))
        .flatMap((chapter) => chapter.topics.map((topic) => topic.id)),
    );
    setChapterIds(nextChapterIds);
    setTopicIds((current) => current.filter((topicId) => validTopicIds.has(topicId)));
  };
  const toggleTopic = (topicId) => {
    if (!topicIds.includes(topicId) && topicIds.length >= MAX_MOCK_TOPICS) {
      setState({ loading: false, result: null, error: new ApiError(`A 25-mark paper can cover at most ${MAX_MOCK_TOPICS} sections. Clear one before adding another.`) });
      return;
    }
    setTopicIds((current) => current.includes(topicId)
      ? current.filter((item) => item !== topicId)
      : [...current, topicId]);
    setState((current) => ({ ...current, error: null }));
  };
  const toggleChapterTopics = (chapter) => {
    const ids = chapter.topics.map((topic) => topic.id);
    const allSelected = ids.every((id) => topicIds.includes(id));
    if (allSelected) {
      setTopicIds((current) => current.filter((id) => !ids.includes(id)));
      return;
    }
    const additions = ids.filter((id) => !topicIds.includes(id));
    if (topicIds.length + additions.length > MAX_MOCK_TOPICS) {
      setState({ loading: false, result: null, error: new ApiError(`This chapter would exceed the ${MAX_MOCK_TOPICS}-section paper limit. Select individual sections instead.`) });
      return;
    }
    setTopicIds((current) => [...new Set([...current, ...ids])]);
    setState((current) => ({ ...current, error: null }));
  };
  const submit = async (event) => {
    event.preventDefault();
    if (!chapterIds.length || !topicIds.length) {
      setState({ loading: false, result: null, error: new ApiError("Select at least one chapter and one topic.") });
      return;
    }
    setState({ loading: true, result: null, error: null });
    try {
      const result = await api.post("/mock-tests", { level, chapter_ids: chapterIds, topic_ids: topicIds, difficulty });
      setState({ loading: false, result, error: null }); onCreated?.();
    } catch (error) {
      setState({ loading: false, result: null, error });
    }
  };
  return (
    <>
      <PageHeader eyebrow="Syllabus-scoped practice" title="Create a 25-mark mock test" description="Choose exact chapters and sections using the page numbers printed in the supplied coursebooks." />
      <div className="mock-test-layout">
        <form className="panel stack-form mock-scope-panel" onSubmit={submit}>
          <div className="panel-heading"><h2>Build the test scope</h2><p>All page ranges below are printed book pages—not PDF viewer indexes.</p></div>
          <ol className="scope-steps" aria-label="Mock-test scope steps">
            <li className="complete"><span>1</span>Level</li>
            <li className={chapterIds.length ? "complete" : "active"}><span>2</span>Chapters</li>
            <li className={topicIds.length ? "complete" : chapterIds.length ? "active" : ""}><span>3</span>Sections</li>
            <li className={topicIds.length ? "active" : ""}><span>4</span>Generate</li>
          </ol>
          <fieldset className="scope-section">
            <legend><span className="section-step">1</span><span><strong>Choose level</strong><small>Load the matching coursebook and syllabus.</small></span></legend>
            <div className="level-card-grid">
              {[["O_LEVEL", "O Level", "10 coursebook chapters"], ["A_LEVEL", "A Level", "30 coursebook chapters"]].map(([value, title, meta]) => (
                <button type="button" key={value} aria-pressed={level === value}
                  className={`level-card ${level === value ? "selected" : ""}`} onClick={() => setLevel(value)}>
                  <span className="level-monogram">{value === "O_LEVEL" ? "O" : "A"}</span>
                  <span><strong>{title} Computer Science</strong><small>{meta}</small></span>
                  <span className="selection-indicator">{level === value ? "✓" : ""}</span>
                </button>
              ))}
            </div>
          </fieldset>
          {syllabus.loading && <p className="muted">Loading syllabus…</p>}
          {syllabus.error && <Notice type="error">{syllabus.error.message}</Notice>}
          {!!chapters.length && (
            <>
              <fieldset className="scope-section">
                <legend><span className="section-step">2</span><span><strong>Select chapter(s)</strong><small>Titles and ranges follow the supplied {level === "O_LEVEL" ? "O Level" : "A Level"} coursebook.</small></span></legend>
                <div className="chapter-card-grid">
                  {chapters.map((chapter) => (
                    <button type="button" aria-pressed={chapterIds.includes(chapter.id)}
                      className={`chapter-card ${chapterIds.includes(chapter.id) ? "selected" : ""}`}
                      key={chapter.id} onClick={() => toggleChapter(chapter.id)}>
                      <span className="chapter-number">Chapter {chapter.chapter_number}</span>
                      <strong>{chapter.name}</strong>
                      <span className="book-pages">{chapter.book_page_label}</span>
                      <small>{chapter.topics.length} section{chapter.topics.length === 1 ? "" : "s"}</small>
                      <span className="chapter-check">{chapterIds.includes(chapter.id) ? "✓" : "+"}</span>
                    </button>
                  ))}
                </div>
              </fieldset>
              {!!availableTopics.length && (
                <fieldset className="scope-section topic-scope">
                  <legend><span className="section-step">3</span><span><strong>Choose exact sections</strong><small>Only selected sections will be assessed.</small></span></legend>
                  <div className="topic-chapter-list">
                    {selectedChapters.map((chapter) => {
                      const selectedCount = chapter.topics.filter((topic) => topicIds.includes(topic.id)).length;
                      const allSelected = selectedCount === chapter.topics.length;
                      return (
                        <section className="topic-chapter-group" key={chapter.id}>
                          <header>
                            <span><small>Chapter {chapter.chapter_number}</small><strong>{chapter.name}</strong></span>
                            <button type="button" className="text-button" onClick={() => toggleChapterTopics(chapter)}>
                              {allSelected ? "Clear chapter" : "Select all"} <span>{selectedCount}/{chapter.topics.length}</span>
                            </button>
                          </header>
                          <div className="topic-card-grid">
                            {chapter.topics.map((topic) => (
                              <button type="button" aria-pressed={topicIds.includes(topic.id)}
                                className={`topic-card ${topicIds.includes(topic.id) ? "selected" : ""}`}
                                disabled={!topicIds.includes(topic.id) && topicIds.length >= MAX_MOCK_TOPICS}
                                key={topic.id} onClick={() => toggleTopic(topic.id)}>
                                <span className="topic-select-box">{topicIds.includes(topic.id) ? "✓" : ""}</span>
                                <span><small>Section {topic.section_number}</small><strong>{topic.name}</strong></span>
                                <span className="book-pages">{topic.book_page_label}</span>
                              </button>
                            ))}
                          </div>
                        </section>
                      );
                    })}
                  </div>
                </fieldset>
              )}
            </>
          )}
          <fieldset className="scope-section generation-settings">
            <legend><span className="section-step">4</span><span><strong>Generation settings</strong><small>Generate an exact 25-mark teacher-review draft.</small></span></legend>
            <div className="generation-row">
              <label>Difficulty<select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}><option value="easy">Easy</option><option value="medium">Medium</option><option value="hard">Hard</option></select></label>
              <div className="scope-summary">
                <span><strong>{chapterIds.length}</strong> chapter{chapterIds.length === 1 ? "" : "s"}</span>
                <span><strong>{topicIds.length}</strong> section{topicIds.length === 1 ? "" : "s"}</span>
                <span><strong>{MAX_MOCK_TOPICS - topicIds.length}</strong> section slots left</span>
                <span><strong>25</strong> marks</span>
              </div>
          <button className="primary-button" disabled={state.loading || syllabus.loading}>{state.loading ? "Generating 25-mark test…" : "Generate teacher-review mock test"}</button>
            </div>
          </fieldset>
          {state.error && <ValidationError error={state.error} />}
        </form>
        <section>
          {state.loading && <Loading label="Planning marks, retrieving evidence, and validating the mock test…" />}
          {state.result ? <AssessmentContent content={state.result.content} includeAnswers /> : !state.loading && <EmptyState title="Select your syllabus scope" description="The exact 25-mark test, answers, rubric, and sources will appear here." />}
        </section>
      </div>
    </>
  );
}

function TeacherAssessments({ api, refreshKey }) {
  const [version, setVersion] = useState(0);
  const state = useAsyncData(() => api.get("/assessments/mine"), [refreshKey, version]);
  const [openId, setOpenId] = useState(null);
  const [message, setMessage] = useState("");

  const mutate = async (path, method, success) => {
    setMessage("");
    try {
      await api[method](path, {});
      setMessage(success);
      setVersion((item) => item + 1);
    } catch (error) { setMessage(error.message); }
  };
  const download = async (assessment, format, includeSolutions) => {
    try {
      const blob = await api.download(`/assessment/${assessment.id}/export/${format}?include_solutions=${includeSolutions}`);
      saveBlob(blob, `${includeSolutions ? "teacher_key" : "student_paper"}_${assessment.id}.${format}`);
    } catch (error) { setMessage(error.message); }
  };
  return (
    <>
      <PageHeader eyebrow="Review before release" title="My assessments" description="Inspect drafts, export student and teacher versions, then publish only when the content is ready." />
      {message && <Notice type={message.includes("failed") || message.includes("cannot") ? "error" : "success"} onClose={() => setMessage("")}>{message}</Notice>}
      {state.loading && <Loading label="Loading assessments…" />}
      {state.error && <Notice type="error">{state.error.message}</Notice>}
      {!state.loading && !state.data?.length && <EmptyState title="No assessments yet" description="Generate a quiz, assignment, or mock test to begin." />}
      <div className="assessment-list">
        {(state.data || []).map((assessment, index) => (
          <article className="assessment-row" key={assessment.id}>
            <button className="assessment-summary" onClick={() => setOpenId(openId === assessment.id ? null : assessment.id)}>
              <span className="list-number">{String(index + 1).padStart(2, "0")}</span>
              <span><strong>{assessment.content?.title || assessment.topic}</strong><small>{assessment.assessment_type?.replaceAll("_", " ")} · {assessment.difficulty}</small></span>
              <span className={`status-pill status-${assessment.status}`}>{assessment.status?.replaceAll("_", " ")}</span>
              <span className="chevron">{openId === assessment.id ? "−" : "+"}</span>
            </button>
            {openId === assessment.id && (
              <div className="assessment-detail">
                <div className="toolbar">
                  <div>
                    <button onClick={() => download(assessment, "pdf", false)}><Icon name="download" /> Student PDF</button>
                    <button onClick={() => download(assessment, "docx", false)}><Icon name="download" /> Student DOCX</button>
                    <button onClick={() => download(assessment, "pdf", true)}><Icon name="download" /> Teacher key</button>
                  </div>
                  {assessment.status !== "approved" && <div><button className="danger-button" onClick={() => mutate(`/assessment/${assessment.id}`, "delete", "Draft deleted.")}><Icon name="delete" /> Delete</button><button className="primary-button compact-button" onClick={() => mutate(`/assessment/${assessment.id}/approve`, "post", "Assessment published to students.")}><Icon name="approve" /> Approve for students</button></div>}
                </div>
                <AssessmentContent content={assessment.content} includeAnswers />
              </div>
            )}
          </article>
        ))}
      </div>
    </>
  );
}

function GradeReview({ api }) {
  const [version, setVersion] = useState(0);
  const state = useAsyncData(() => api.get("/grades/pending-review"), [version]);
  const [forms, setForms] = useState({});
  const review = async (grade) => {
    const form = forms[grade.id] || { score: grade.ai_score || 0, comments: "" };
    try {
      await api.post(`/grade/${grade.id}/review`, { human_score: Number(form.score), comments: form.comments || null });
      setVersion((item) => item + 1);
    } catch (error) { window.alert(error.message); }
  };
  return (
    <>
      <PageHeader eyebrow="Human-in-the-loop grading" title="Grade review" description="AI grading is a draft. Inspect the evidence and record the final teacher score." />
      {state.loading && <Loading label="Loading grading drafts…" />}
      {state.error && <Notice type="error">{state.error.message}</Notice>}
      {!state.loading && !state.data?.length && <EmptyState title="Nothing awaiting review" description="New student submissions with AI grading drafts will appear here." />}
      <div className="review-grid">
        {(state.data || []).map((grade) => {
          const form = forms[grade.id] || { score: grade.ai_score || 0, comments: "" };
          return <article className="panel" key={grade.id}><div className="grade-score"><span>AI draft</span><strong>{grade.ai_score} / {grade.max_score}</strong></div><p>{grade.comments}</p><details><summary>View grading trace</summary><pre className="diagnostics">{JSON.stringify(grade.details, null, 2)}</pre></details><label>Final teacher score<input type="number" min="0" max={grade.max_score} step="0.5" value={form.score} onChange={(e) => setForms({ ...forms, [grade.id]: { ...form, score: e.target.value } })} /></label><label>Teacher comments<textarea rows="3" value={form.comments} onChange={(e) => setForms({ ...forms, [grade.id]: { ...form, comments: e.target.value } })} /></label><button className="primary-button" onClick={() => review(grade)}>Confirm final score</button></article>;
        })}
      </div>
    </>
  );
}

function StudentAssessments({ api }) {
  const state = useAsyncData(() => api.get("/assessments/available"), []);
  const [openId, setOpenId] = useState(null);
  const [answers, setAnswers] = useState({});
  const [messages, setMessages] = useState({});
  const submit = async (assessment) => {
    const values = answers[assessment.id] || {};
    const answerText = (assessment.content?.questions || []).map((q, i) => `Q${q.number || i + 1}: ${values[q.number || i + 1] || ""}`).join("\n");
    if (!answerText.replace(/Q\d+:\s*/g, "").trim()) {
      setMessages({ ...messages, [assessment.id]: "Answer at least one question before submitting." }); return;
    }
    try {
      const result = await api.post(`/assessment/${assessment.id}/submissions`, { answer_text: answerText });
      setMessages({ ...messages, [assessment.id]: `Submission saved. Status: ${result.grading?.evaluation?.status || "teacher review pending"}.` });
    } catch (error) { setMessages({ ...messages, [assessment.id]: error.message }); }
  };
  return (
    <>
      <PageHeader eyebrow="Published by your teachers" title="Assessments" description="Complete approved work here. Your submission is saved for teacher-reviewed grading." />
      {state.loading && <Loading label="Loading available assessments…" />}
      {state.error && <Notice type="error">{state.error.message}</Notice>}
      {!state.loading && !state.data?.length && <EmptyState title="No assessments available" description="Your teacher has not published an assessment yet." />}
      <div className="assessment-list">
        {(state.data || []).map((assessment, index) => (
          <article className="assessment-row" key={assessment.id}>
            <button className="assessment-summary" onClick={() => setOpenId(openId === assessment.id ? null : assessment.id)}><span className="list-number">{String(index + 1).padStart(2, "0")}</span><span><strong>{assessment.content?.title || assessment.topic}</strong><small>{assessment.assessment_type?.replaceAll("_", " ")} · {assessment.difficulty}</small></span><span className="status-pill status-approved">Available</span><span className="chevron">{openId === assessment.id ? "−" : "+"}</span></button>
            {openId === assessment.id && <div className="assessment-detail"><AssessmentContent content={assessment.content} answerInputs={answers[assessment.id] || {}} onAnswer={(number, value) => setAnswers({ ...answers, [assessment.id]: { ...(answers[assessment.id] || {}), [number]: value } })} />{messages[assessment.id] && <Notice type={messages[assessment.id].startsWith("Submission saved") ? "success" : "error"}>{messages[assessment.id]}</Notice>}<button className="primary-button" onClick={() => submit(assessment)}>Submit for teacher review</button></div>}
          </article>
        ))}
      </div>
    </>
  );
}

function StudentGrades({ api }) {
  const state = useAsyncData(() => api.get("/student/grades"), []);
  return (
    <>
      <PageHeader eyebrow="Progress and feedback" title="My grades" description="See AI drafts and final teacher-reviewed results without losing the grading audit trail." />
      {state.loading && <Loading label="Loading grades…" />}
      {state.error && <Notice type="error">{state.error.message}</Notice>}
      {!state.loading && !state.data?.length && <EmptyState title="No grades yet" description="Results will appear after you submit an assessment." />}
      <div className="grade-table-wrap">
        {!!state.data?.length && <table className="grade-table"><thead><tr><th>Submission</th><th>AI draft</th><th>Final score</th><th>Maximum</th><th>Status</th><th>Feedback</th></tr></thead><tbody>{state.data.map((grade) => <tr key={grade.id}><td>#{grade.id}</td><td>{grade.ai_score ?? "—"}</td><td><strong>{grade.human_score ?? "Pending"}</strong></td><td>{grade.max_score}</td><td><span className={`status-pill ${grade.human_score == null ? "status-pending_review" : "status-approved"}`}>{grade.human_score == null ? "Teacher review" : "Reviewed"}</span></td><td>{grade.comments || "—"}</td></tr>)}</tbody></table>}
      </div>
    </>
  );
}

function Monitoring({ api }) {
  const state = useAsyncData(() => api.get("/monitoring/summary?days=30"), []);
  const telemetry = state.data?.local_telemetry || {};
  const download = async () => {
    try { saveBlob(await api.download("/monitoring/live-workbook"), "live_answers.xlsx"); }
    catch (error) { window.alert(error.message); }
  };
  return (
    <>
      <PageHeader eyebrow="Operational visibility" title="RAG monitoring" description="Inspect recent system usage and download the append-only live answer workbook." action={<button className="secondary-button" onClick={download}><Icon name="download" /> Download workbook</button>} />
      {state.loading && <Loading label="Loading telemetry…" />}
      {state.error && <Notice type="error">{state.error.message}</Notice>}
      {state.data && <><div className="stats-grid"><StatCard label="Requests" value={telemetry.request_count ?? telemetry.count ?? 0} /><StatCard label="Median latency" value={telemetry.median_latency_ms ? `${Math.round(telemetry.median_latency_ms)} ms` : "—"} /><StatCard label="Technical failures" value={telemetry.technical_failure_count ?? 0} /><StatCard label="LangSmith tracing" value={state.data.langsmith?.enabled ? "Enabled" : "Disabled"} /></div><div className="panel"><h2>Telemetry snapshot</h2><pre className="diagnostics">{JSON.stringify(state.data, null, 2)}</pre></div></>}
    </>
  );
}

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(Boolean(token));
  const [active, setActive] = useState("home");
  const [menuOpen, setMenuOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [initialQuestion, setInitialQuestion] = useState("");

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null); setUser(null); setActive("home");
  }, []);
  const api = useMemo(() => new ApiClient(() => localStorage.getItem(TOKEN_KEY), logout), [logout]);

  useEffect(() => {
    if (!token) { setAuthLoading(false); return; }
    api.get("/auth/me").then(setUser).catch(logout).finally(() => setAuthLoading(false));
  }, [api, logout, token]);

  const authenticated = (profile) => {
    setToken(localStorage.getItem(TOKEN_KEY)); setUser(profile); setActive("home");
  };
  const navigate = (page, question = "") => {
    setActive(page);
    if (question) setInitialQuestion(question);
  };

  if (authLoading) return <div className="boot-screen"><span className="spinner" /><p>Opening your workspace…</p></div>;
  if (!user) return <AuthScreen api={api} onAuthenticated={authenticated} />;

  let content;
  if (active === "home") content = user.role === "teacher" ? <TeacherHome api={api} navigate={navigate} /> : <StudentHome api={api} navigate={navigate} />;
  else if (active === "rag") content = <RagAssistant api={api} role={user.role} initialQuestion={initialQuestion} />;
  else if (active === "generate" && user.role === "teacher") content = <AssessmentGenerator api={api} onCreated={() => setRefreshKey((key) => key + 1)} />;
  else if (active === "mock-test" && user.role === "teacher") content = <MockTestGenerator api={api} onCreated={() => setRefreshKey((key) => key + 1)} />;
  else if (active === "assessments") content = user.role === "teacher" ? <TeacherAssessments api={api} refreshKey={refreshKey} /> : <StudentAssessments api={api} />;
  else if (active === "grade-review" && user.role === "teacher") content = <GradeReview api={api} />;
  else if (active === "grades" && user.role === "student") content = <StudentGrades api={api} />;
  else if (active === "monitoring" && user.role === "teacher") content = <Monitoring api={api} />;
  else content = <EmptyState title="Page unavailable" description="This feature is not available for your account role." />;

  return (
    <div className="app-shell">
      <Sidebar user={user} active={active} onNavigate={navigate} onLogout={logout} open={menuOpen} onClose={() => setMenuOpen(false)} />
      <div className="app-main">
        <Topbar user={user} onMenu={() => setMenuOpen(true)} />
        <main className="content">{content}</main>
      </div>
    </div>
  );
}
