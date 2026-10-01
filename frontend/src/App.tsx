import { useEffect, useRef, useState } from "react";
import { api, subscribeToRun } from "./api";
import { isComplete, newExample } from "./components/ExamplesEditor";
import { LoopIcon, XIcon } from "./components/Icons";
import { Results } from "./components/Results";
import { ReviewStep } from "./components/ReviewStep";
import { RunProgress } from "./components/RunProgress";
import { SetupStep } from "./components/SetupStep";
import { Stepper } from "./components/Stepper";
import type {
  AppConfig,
  EditableExample,
  Example,
  LineageStep,
  ModelAccess,
  RunEvent,
  RunResult,
  RunSettings,
} from "./types";

type Step = "setup" | "review" | "running" | "done";
const STEPS: { id: Step; label: string }[] = [
  { id: "setup", label: "Task" },
  { id: "review", label: "Examples" },
  { id: "running", label: "Optimise" },
  { id: "done", label: "Results" },
];
// The API accepts at most this many examples per generation request.
const MAX_GENERATE = 30;
const GENERATE_MORE = 5;

const errorMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [step, setStep] = useState<Step>("setup");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [task, setTask] = useState("");
  // Model and key live only in memory: a page reload deliberately forgets the key.
  const [access, setAccess] = useState<ModelAccess>({ model: "", apiKey: "" });
  const [examples, setExamples] = useState<EditableExample[]>([newExample()]);
  const [settings, setSettings] = useState<RunSettings>({
    scorer: "auto",
    rubric: "",
    nCandidates: 3,
    maxIterations: 4,
  });

  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [lineage, setLineage] = useState<LineageStep[]>([]);
  const unsubscribe = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setConfig(c);
        setSettings((s) => ({
          ...s,
          nCandidates: c.default_candidates,
          maxIterations: Math.min(c.default_max_iterations, 8),
        }));
      })
      .catch((e) => setError(`Could not reach the server: ${errorMessage(e)}`));
    return () => unsubscribe.current?.();
  }, []);

  // Each step is a new page as far as the user is concerned: start it at the top.
  useEffect(() => window.scrollTo({ top: 0 }), [step]);

  const completeExamples = (): Example[] =>
    examples.filter(isComplete).map((e) => ({
      input: e.input.trim(),
      expected_output: e.expected_output.trim(),
    }));

  async function generate(n: number) {
    if (!config) return;
    const room = config.max_examples - completeExamples().length;
    const count = Math.min(n, room, MAX_GENERATE);
    if (count <= 0) return;
    setBusy(true);
    setError(null);
    try {
      const { examples: fresh } = await api.generateExamples(
        access,
        task.trim(),
        completeExamples(),
        count,
      );
      setExamples((prev) => [
        ...prev.filter(isComplete),
        ...fresh.map((e) => newExample(e.input, e.expected_output, true)),
      ]);
      setStep("review");
      if (fresh.length === 0) {
        setError("The model returned no new examples. Try again, or add some yourself.");
      }
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  function onEvent(event: RunEvent) {
    // EventSource may redeliver after a reconnect; seq makes that idempotent.
    setEvents((prev) => (prev.some((p) => p.seq === event.seq) ? prev : [...prev, event]));
    if (event.type === "result") {
      setResult(event.data.result as RunResult);
      setLineage(event.data.lineage as LineageStep[]);
      setStep("done");
    } else if (event.type === "error") {
      setRunError(event.message);
    } else if (event.type === "cancelled") {
      setRunError("Run cancelled.");
    }
  }

  async function start() {
    setBusy(true);
    setError(null);
    setRunError(null);
    setEvents([]);
    setResult(null);
    try {
      const created = await api.createRun(access, task.trim(), completeExamples(), settings);
      setRunId(created.run_id);
      setStep("running");
      unsubscribe.current?.();
      unsubscribe.current = subscribeToRun(created.run_id, onEvent, () =>
        setRunError("Lost connection to the server."),
      );
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!runId) return;
    try {
      await api.cancelRun(runId);
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  function startOver() {
    unsubscribe.current?.();
    setTask("");
    setExamples([newExample()]);
    setEvents([]);
    setResult(null);
    setRunError(null);
    setStep("setup");
  }

  const stepIndex = STEPS.findIndex((s) => s.id === step);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">
            <LoopIcon size={18} />
          </span>
          PromptLoop
        </div>
        <a className="muted small" href="/docs" target="_blank" rel="noreferrer">
          API
        </a>
      </header>

      <main className="container">
        {step === "setup" && (
          <div className="intro">
            <h1>
              Find the prompt that <span className="accent-text">actually works</span>.
            </h1>
            <p>
              Describe a task and show one good answer. PromptLoop writes candidate prompts, tests
              them on your model, learns from the failures, and proves the winner on examples it
              never saw.
            </p>
          </div>
        )}
        <Stepper steps={STEPS.map((s) => s.label)} current={stepIndex} />

        {error && (
          <div className="toast" role="alert">
            <span>{error}</span>
            <button type="button" className="icon-btn" onClick={() => setError(null)} aria-label="Dismiss">
              <XIcon />
            </button>
          </div>
        )}

        {!config ? (
          !error && (
            <p className="status center">
              <span className="spinner" /> Loading…
            </p>
          )
        ) : step === "setup" ? (
          <SetupStep
            config={config}
            task={task}
            onTaskChange={setTask}
            access={access}
            onAccessChange={setAccess}
            examples={examples}
            onExamplesChange={setExamples}
            busy={busy}
            onGenerate={() =>
              generate(Math.max(config.target_total_examples - completeExamples().length, 3))
            }
            onSkip={() => setStep("review")}
          />
        ) : step === "review" ? (
          <ReviewStep
            config={config}
            examples={examples}
            onExamplesChange={setExamples}
            settings={settings}
            onSettingsChange={setSettings}
            busy={busy}
            onGenerateMore={() => generate(GENERATE_MORE)}
            onBack={() => setStep("setup")}
            onStart={start}
          />
        ) : step === "running" ? (
          <RunProgress
            events={events}
            maxRounds={settings.maxIterations}
            error={runError}
            onCancel={cancel}
            onBack={() => setStep("review")}
          />
        ) : (
          result && (
            <Results
              result={result}
              lineage={lineage}
              onEditAndRerun={() => setStep("review")}
              onStartOver={startOver}
            />
          )
        )}
      </main>

      <footer className="footer">
        Open source · runs on your own model and key · nothing you enter is stored
      </footer>
    </div>
  );
}
