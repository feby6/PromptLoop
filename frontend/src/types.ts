// Mirrors the backend's Pydantic models (src/promptloop/models.py, api/schemas.py).
// Keep the two in sync when either side changes.

export type ScorerName = "exact" | "json_match" | "judge";
export type ScorerChoice = "auto" | ScorerName;

export interface Example {
  input: string;
  expected_output: string;
}

/** An example row in the editor; `id` is a stable React key, never sent to the API. */
export interface EditableExample extends Example {
  id: string;
  generated: boolean;
}

export interface Candidate {
  id: string;
  prompt: string;
  iteration: number;
  parent_ids: string[];
}

export interface ExampleResult {
  input: string;
  expected_output: string;
  output: string;
  score: number;
  error: string | null;
}

export interface EvalResult {
  candidate_id: string;
  results: ExampleResult[];
  mean_score: number;
}

export interface IterationRecord {
  iteration: number;
  candidates: Candidate[];
  evals: EvalResult[];
  best_candidate_id: string;
  best_score: number;
  critique: string | null;
}

export interface LLMUsage {
  calls: number;
  cache_hits: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export type StopReason = "target_reached" | "max_iterations" | "plateau";

export interface RunResult {
  task_name: string;
  target_model: string;
  optimizer_model: string;
  scorer: ScorerName;
  best_candidate: Candidate;
  train_score: number;
  val_score: number;
  val_eval: EvalResult;
  baseline_train_score: number | null;
  baseline_val_score: number | null;
  stop_reason: StopReason;
  history: IterationRecord[];
  n_train: number;
  n_val: number;
  usage: LLMUsage;
}

export interface DiffChunk {
  op: "equal" | "insert" | "delete";
  text: string;
}

export interface LineageStep {
  candidate: Candidate;
  train_score: number | null;
  diff_from_parent: DiffChunk[];
}

export type EventType = "phase" | "candidates" | "iteration" | "result" | "error" | "cancelled";
export const EVENT_TYPES: EventType[] = [
  "phase",
  "candidates",
  "iteration",
  "result",
  "error",
  "cancelled",
];
export const TERMINAL_EVENTS: ReadonlySet<EventType> = new Set(["result", "error", "cancelled"]);

export interface RunEvent {
  seq: number;
  type: EventType;
  message: string;
  data: Record<string, unknown>;
}

export interface AppConfig {
  suggested_models: string[];
  min_examples: number;
  target_total_examples: number;
  max_examples: number;
  default_candidates: number;
  default_max_iterations: number;
}

export interface RunCreated {
  run_id: string;
  scorer: ScorerName;
  n_train: number;
  n_val: number;
}

export interface RunSettings {
  scorer: ScorerChoice;
  rubric: string;
  nCandidates: number;
  maxIterations: number;
}

/** Model + key, held only in React state: never written to storage. */
export interface ModelAccess {
  model: string;
  apiKey: string;
}
