import type { EditableExample } from "../types";
import { AutoTextarea } from "./AutoTextarea";
import { PlusIcon, TrashIcon } from "./Icons";

let nextId = 0;
export function newExample(
  input = "",
  expected_output = "",
  generated = false,
): EditableExample {
  nextId += 1;
  return { id: `ex-${nextId}`, input, expected_output, generated };
}

export function isComplete(e: EditableExample): boolean {
  return e.input.trim() !== "" && e.expected_output.trim() !== "";
}

interface Props {
  examples: EditableExample[];
  onChange: (examples: EditableExample[]) => void;
  maxExamples: number;
}

/** Input → expected-output pairs as a compact two-column table. Generated rows are
 *  marked until edited, because a model can get its own expected outputs wrong. */
export function ExamplesEditor({ examples, onChange, maxExamples }: Props) {
  const update = (id: string, patch: Partial<EditableExample>) =>
    onChange(examples.map((e) => (e.id === id ? { ...e, ...patch } : e)));
  const remove = (id: string) => onChange(examples.filter((e) => e.id !== id));

  return (
    <div className="examples">
      <div className="examples-head" aria-hidden>
        <span />
        <span>Input</span>
        <span>Expected output</span>
        <span />
      </div>
      {examples.map((e, i) => (
        <div key={e.id} className={`example-row${e.generated ? " generated" : ""}`}>
          <span className="example-index" title={e.generated ? "Generated: please check" : ""}>
            {e.generated ? <span className="gen-dot" /> : null}
            {i + 1}
          </span>
          <AutoTextarea
            aria-label={`Example ${i + 1} input`}
            value={e.input}
            onChange={(ev) => update(e.id, { input: ev.target.value })}
            placeholder="What the model receives"
          />
          <AutoTextarea
            aria-label={`Example ${i + 1} expected output`}
            className="mono"
            value={e.expected_output}
            // Editing a generated example means the user has vetted it.
            onChange={(ev) => update(e.id, { expected_output: ev.target.value, generated: false })}
            placeholder="The exact answer you want"
          />
          <button
            type="button"
            className="icon-btn"
            onClick={() => remove(e.id)}
            disabled={examples.length <= 1}
            aria-label={`Remove example ${i + 1}`}
            title="Remove"
          >
            <TrashIcon />
          </button>
        </div>
      ))}
      <button
        type="button"
        className="add-row"
        onClick={() => onChange([...examples, newExample()])}
        disabled={examples.length >= maxExamples}
      >
        <PlusIcon /> Add example
      </button>
    </div>
  );
}
