import { useState } from "react";
import { SAMPLES } from "../samples";
import type { AppConfig, EditableExample, ModelAccess } from "../types";
import { ExamplesEditor, isComplete, newExample } from "./ExamplesEditor";
import { EyeIcon, EyeOffIcon, LockIcon, SparkleIcon } from "./Icons";

interface Props {
  config: AppConfig;
  task: string;
  onTaskChange: (task: string) => void;
  access: ModelAccess;
  onAccessChange: (access: ModelAccess) => void;
  examples: EditableExample[];
  onExamplesChange: (examples: EditableExample[]) => void;
  busy: boolean;
  onGenerate: () => void;
  onSkip: () => void;
}

// LiteLLM provider prefixes. "Other" lets users type any full LiteLLM model string.
const PROVIDERS = [
  { id: "groq", label: "Groq" },
  { id: "openai", label: "OpenAI" },
  { id: "anthropic", label: "Anthropic" },
  { id: "gemini", label: "Gemini" },
  { id: "openrouter", label: "OpenRouter" },
  { id: "", label: "Other" },
];
const MODEL_PATTERN = /^[A-Za-z0-9_-]+\/\S+$/;

/** Split "groq/qwen/qwen3" into provider "groq" and name "qwen/qwen3". */
function splitModel(model: string): { provider: string; name: string } {
  const slash = model.indexOf("/");
  const prefix = slash > 0 ? model.slice(0, slash) : "";
  if (PROVIDERS.some((p) => p.id === prefix && p.id !== "")) {
    return { provider: prefix, name: model.slice(slash + 1) };
  }
  return { provider: "", name: model };
}

export function SetupStep(props: Props) {
  const { config, task, access, examples, busy } = props;
  const [showKey, setShowKey] = useState(false);
  const { provider, name } = splitModel(access.model);
  const [chosenProvider, setChosenProvider] = useState(provider || "groq");

  const setModelName = (value: string) =>
    props.onAccessChange({
      ...access,
      model: chosenProvider ? `${chosenProvider}/${value}` : value,
    });
  const chooseProvider = (id: string) => {
    setChosenProvider(id);
    props.onAccessChange({ ...access, model: id && name ? `${id}/${name}` : name });
  };
  const suggestions = config.suggested_models
    .filter((m) => chosenProvider && m.startsWith(`${chosenProvider}/`))
    .map((m) => m.slice(chosenProvider.length + 1));

  const loadSample = (index: number) => {
    const sample = SAMPLES[index];
    props.onTaskChange(sample.task);
    props.onExamplesChange(sample.examples.map((e) => newExample(e.input, e.expected_output)));
  };

  const complete = examples.filter(isComplete);
  const modelValid = MODEL_PATTERN.test(access.model.trim());
  const ready = task.trim().length >= 10 && modelValid && access.apiKey.trim() !== "";
  const canGenerate = ready && complete.length >= 1;
  const canSkip = ready && complete.length >= config.min_examples;

  return (
    <div className="panel">
      <section className="section">
        <div className="section-head">
          <div>
            <h2>What should the prompt do?</h2>
            <p className="muted">Describe the task in a sentence or two, as you would to a colleague.</p>
          </div>
          <div className="samples">
            <span className="muted small">Try a sample:</span>
            {SAMPLES.map((s, i) => (
              <button key={s.label} type="button" className="chip" onClick={() => loadSample(i)}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
        <textarea
          className="task-input"
          rows={3}
          value={task}
          onChange={(e) => props.onTaskChange(e.target.value)}
          placeholder="e.g. Extract the vendor, date and total from an invoice and return them as JSON."
        />
      </section>

      <section className="section">
        <h2>Which model will use it?</h2>
        <p className="muted">
          Everything runs on your model with your key: writing, testing and improving the prompts.
        </p>
        <div className="segmented" role="radiogroup" aria-label="Provider">
          {PROVIDERS.map((p) => (
            <button
              key={p.label}
              type="button"
              role="radio"
              aria-checked={chosenProvider === p.id}
              className={chosenProvider === p.id ? "active" : ""}
              onClick={() => chooseProvider(p.id)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="grid-2">
          <label className="field">
            <span>Model</span>
            <div className="input-affix">
              {chosenProvider && <span className="affix">{chosenProvider}/</span>}
              <input
                list="model-suggestions"
                value={chosenProvider ? name : access.model}
                onChange={(e) => setModelName(e.target.value)}
                placeholder={chosenProvider ? "model name" : "provider/model (LiteLLM format)"}
                spellCheck={false}
              />
            </div>
            <datalist id="model-suggestions">
              {suggestions.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
          <label className="field">
            <span>API key</span>
            <div className="input-affix">
              <input
                type={showKey ? "text" : "password"}
                autoComplete="off"
                spellCheck={false}
                value={access.apiKey}
                onChange={(e) => props.onAccessChange({ ...access, apiKey: e.target.value })}
                placeholder="Paste your key"
              />
              <button
                type="button"
                className="icon-btn"
                onClick={() => setShowKey((v) => !v)}
                aria-label={showKey ? "Hide key" : "Show key"}
              >
                {showKey ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </label>
        </div>
        <p className="note">
          <LockIcon size={14} /> Your key stays in this tab and is sent only to run your model. It
          is never stored or logged, and reloading the page forgets it.
        </p>
      </section>

      <section className="section">
        <h2>Show what a good answer looks like</h2>
        <p className="muted">
          One example is enough to start; we'll draft more for you to check. A run needs at
          least {config.min_examples}, and about a third are held back to score the result fairly.
        </p>
        <ExamplesEditor
          examples={examples}
          onChange={props.onExamplesChange}
          maxExamples={config.max_examples}
        />
      </section>

      <div className="panel-actions">
        <button
          type="button"
          className="btn ghost"
          onClick={props.onSkip}
          disabled={!canSkip || busy}
          title={canSkip ? "" : `Needs at least ${config.min_examples} complete examples`}
        >
          I have enough examples
        </button>
        <button
          type="button"
          className="btn primary"
          onClick={props.onGenerate}
          disabled={!canGenerate || busy}
        >
          {busy ? <span className="spinner" /> : <SparkleIcon />}
          {busy ? "Drafting examples…" : "Draft more examples"}
        </button>
      </div>
    </div>
  );
}
