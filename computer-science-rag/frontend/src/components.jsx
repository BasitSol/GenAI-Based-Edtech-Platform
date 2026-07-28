import React from "react";

export function Icon({ name }) {
  const icons = {
    home: "⌂",
    chat: "✦",
    assessment: "▤",
    mock: "◇",
    grades: "✓",
    library: "▦",
    monitoring: "◉",
    logout: "↗",
    menu: "☰",
    close: "×",
    user: "●",
    download: "↓",
    approve: "✓",
    delete: "×",
  };
  return <span className="icon" aria-hidden="true">{icons[name] || "•"}</span>;
}

export function PageHeader({ eyebrow, title, description, action }) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {action}
    </header>
  );
}

export function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <span className="empty-symbol">◇</span>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

export function Loading({ label = "Working on it…" }) {
  return (
    <div className="loading-panel" role="status">
      <span className="spinner" />
      <div>
        <strong>{label}</strong>
        <p>Keep this page open. Complex generation can take several minutes.</p>
      </div>
    </div>
  );
}

export function Notice({ type = "info", children, onClose }) {
  return (
    <div className={`notice notice-${type}`} role={type === "error" ? "alert" : "status"}>
      <span>{children}</span>
      {onClose && <button className="icon-button" onClick={onClose} aria-label="Dismiss">×</button>}
    </div>
  );
}

function parseTable(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length >= 2 && lines.every((line) => line.includes("|"))) {
    const rows = lines
      .filter((line) => !/^\|?[\s:-]+(?:\|[\s:-]+)+\|?$/.test(line))
      .map((line) => line.replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
    if (rows.length >= 2 && rows.every((row) => row.length === rows[0].length)) return rows;
  }
  return null;
}

export function RichText({ value }) {
  if (!value) return null;
  const text = String(value).replace(/\\n/g, "\n");
  const table = parseTable(text);
  if (table) {
    return (
      <div className="table-wrap">
        <table className="content-table">
          <thead><tr>{table[0].map((cell, index) => <th key={index}>{cell}</th>)}</tr></thead>
          <tbody>
            {table.slice(1).map((row, rowIndex) => (
              <tr key={rowIndex}>{row.map((cell, index) => <td key={index}>{cell}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="rich-text">
      {blocks.map((block, index) => {
        const trimmed = block.trim();
        if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
          return <pre key={index}><code>{trimmed.replace(/^```[a-z]*\n?/i, "").replace(/```$/, "")}</code></pre>;
        }
        const lines = trimmed.split("\n");
        if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
          return <ul key={index}>{lines.map((line, i) => <li key={i}>{line.replace(/^\s*[-*]\s+/, "")}</li>)}</ul>;
        }
        if (lines.every((line) => /^\s*\d+[.)]\s+/.test(line))) {
          return <ol key={index}>{lines.map((line, i) => <li key={i}>{line.replace(/^\s*\d+[.)]\s+/, "")}</li>)}</ol>;
        }
        return <p key={index}>{lines.map((line, i) => <React.Fragment key={i}>{line}{i < lines.length - 1 && <br />}</React.Fragment>)}</p>;
      })}
    </div>
  );
}

export function AssessmentContent({ content, includeAnswers = false, answerInputs, onAnswer }) {
  const questions = content?.questions || [];
  return (
    <div className="assessment-paper">
      <div className="paper-heading">
        <p className="eyebrow">Computer Science Assessment</p>
        <h2>{content?.title || "Assessment"}</h2>
        <RichText value={content?.instructions} />
      </div>
      <div className="question-list">
        {questions.map((question, index) => {
          const number = question.number || index + 1;
          const type = String(question.question_type || "SHORT_ANSWER").toUpperCase();
          const options = question.options || [];
          return (
            <article className="question-card" key={`${number}-${index}`}>
              <div className="question-number">Q{number}</div>
              <div className="question-body">
                <RichText value={question.question} />
                <span className="question-meta">{question.marks || 0} mark(s) · {type.replaceAll("_", " ")}</span>
                {type === "MCQ" && (
                  <div className="option-list">
                    {options.map((option, optionIndex) => (
                      <label className="answer-option" key={optionIndex}>
                        {onAnswer ? (
                          <input
                            type="radio"
                            name={`q-${number}`}
                            checked={answerInputs?.[number] === option}
                            onChange={() => onAnswer(number, option)}
                          />
                        ) : <span className="option-letter">{String.fromCharCode(65 + optionIndex)}</span>}
                        <span><b>{String.fromCharCode(65 + optionIndex)}.</b> {option}</span>
                      </label>
                    ))}
                  </div>
                )}
                {onAnswer && type !== "MCQ" && (
                  <textarea
                    className="answer-box"
                    rows={type === "LONG_ANSWER" ? 7 : 4}
                    placeholder="Write your answer here…"
                    value={answerInputs?.[number] || ""}
                    onChange={(event) => onAnswer(number, event.target.value)}
                  />
                )}
                {includeAnswers && (
                  <div className="answer-key">
                    <h4>Model answer</h4>
                    <RichText value={question.model_answer} />
                    {type === "MCQ" && question.correct_option && (
                      <p><strong>Correct option:</strong> {question.correct_option}</p>
                    )}
                    <h4>Marking guidance</h4>
                    <ul>{(question.rubric || []).map((point, i) => <li key={i}>{point}</li>)}</ul>
                    {!!question.citations?.length && (
                      <p className="source-line">
                        Evidence: {question.citations.map((item) => `${item.document_id} p.${item.page}`).join(" · ")}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export function AnswerPanel({ result }) {
  if (!result) return null;
  const citations = result.citations || result.sources || [];
  return (
    <article className="answer-panel">
      <div className="answer-status">
        <span>{String(result.answer_type || "Grounded answer").replaceAll("_", " ")}</span>
        <span>{result.difficulty || "Adaptive"}</span>
        <span>{result.execution_status || "Completed"}</span>
      </div>
      <RichText value={result.answer || result.response} />
      {!!citations.length && (
        <details>
          <summary>View sources ({citations.length})</summary>
          <div className="source-grid">
            {citations.map((source, index) => (
              <div className="source-card" key={source.chunk_id || index}>
                <strong>{source.document_id || source.title || "Source"}</strong>
                <span>Page {source.page || "—"}</span>
                {source.relationship && <small>{source.relationship.replaceAll("_", " ")}</small>}
              </div>
            ))}
          </div>
        </details>
      )}
    </article>
  );
}
