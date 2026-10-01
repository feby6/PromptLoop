import type {
  AppConfig,
  Example,
  ModelAccess,
  RunCreated,
  RunEvent,
  RunSettings,
} from "./types";
import { EVENT_TYPES, TERMINAL_EVENTS } from "./types";

// Empty in production (same origin as the API). Set VITE_API_BASE when the frontend is
// hosted separately from the backend.
const BASE: string = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

interface ValidationIssue {
  loc: (string | number)[];
  msg: string;
}

/** FastAPI returns `detail` as a string for our errors and as a list for validation errors. */
function describeError(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return (detail as ValidationIssue[])
      .map((d) => `${d.loc.slice(1).join(".")}: ${d.msg}`)
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      message = describeError((await res.json()).detail, message);
    } catch {
      // Non-JSON error body: keep the status line.
    }
    throw new ApiError(res.status, message);
  }
  return (await res.json()) as T;
}

export const api = {
  config: () => request<AppConfig>("/api/config"),

  generateExamples: (
    access: ModelAccess,
    taskDescription: string,
    seeds: Example[],
    n: number,
  ) =>
    request<{ examples: Example[] }>("/api/examples/generate", {
      method: "POST",
      body: JSON.stringify({
        model: access.model,
        api_key: access.apiKey,
        task_description: taskDescription,
        seed_examples: seeds,
        n,
      }),
    }),

  createRun: (
    access: ModelAccess,
    taskDescription: string,
    examples: Example[],
    settings: RunSettings,
  ) =>
    request<RunCreated>("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        model: access.model,
        api_key: access.apiKey,
        task_description: taskDescription,
        examples,
        scorer: settings.scorer,
        rubric: settings.rubric.trim() || null,
        n_candidates: settings.nCandidates,
        max_iterations: settings.maxIterations,
      }),
    }),

  cancelRun: (runId: string) =>
    request<unknown>(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" }),
};

/**
 * Stream a run's events. EventSource reconnects on its own after network blips and
 * sends Last-Event-ID, which the server uses to resume without duplicates.
 * Returns an unsubscribe function.
 */
export function subscribeToRun(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onConnectionLost: () => void,
): () => void {
  const source = new EventSource(`${BASE}/api/runs/${encodeURIComponent(runId)}/events`);
  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (msg) => {
      const event = JSON.parse((msg as MessageEvent<string>).data) as RunEvent;
      onEvent(event);
      if (TERMINAL_EVENTS.has(event.type)) source.close();
    });
  }
  source.onerror = () => {
    // CONNECTING means the browser is already retrying; CLOSED means it gave up.
    if (source.readyState === EventSource.CLOSED) onConnectionLost();
  };
  return () => source.close();
}
