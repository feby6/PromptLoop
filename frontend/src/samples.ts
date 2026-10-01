// One-click starting points for first-time visitors. Each has just enough examples to
// show the format; "Generate more examples" fills in the rest.
import type { Example } from "./types";

export interface Sample {
  label: string;
  task: string;
  examples: Example[];
}

export const SAMPLES: Sample[] = [
  {
    label: "Sentiment",
    task: "Classify the sentiment of a product review as positive, negative, or neutral.",
    examples: [
      { input: "Battery lasts two full days and the screen is gorgeous.", expected_output: "positive" },
      { input: "Stopped charging after a week. Support never replied.", expected_output: "negative" },
      { input: "It's a kettle. It boils water.", expected_output: "neutral" },
    ],
  },
  {
    label: "Invoice → JSON",
    task: "Extract the vendor, invoice number, date, total and currency from invoice text and return them as JSON.",
    examples: [
      {
        input: "INVOICE #A-1043\nAcme Tools Ltd, Leeds\nDate: 03/04/2025\nTOTAL DUE £1,440.00",
        expected_output:
          '{"vendor": "Acme Tools Ltd", "invoice_number": "A-1043", "date": "2025-04-03", "total": 1440.0, "currency": "GBP"}',
      },
      {
        input: "Receipt from Blue Bottle Coffee, order 88213, March 9, 2025, total paid $13.50",
        expected_output:
          '{"vendor": "Blue Bottle Coffee", "invoice_number": "88213", "date": "2025-03-09", "total": 13.5, "currency": "USD"}',
      },
    ],
  },
  {
    label: "Ticket triage",
    task: "Route a customer support message to one team: billing, technical, account, or other.",
    examples: [
      { input: "I was charged twice for my subscription this month.", expected_output: "billing" },
      { input: "The app crashes whenever I upload a photo.", expected_output: "technical" },
      { input: "How do I change the email on my profile?", expected_output: "account" },
    ],
  },
];
