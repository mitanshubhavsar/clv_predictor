export type Interval = { value: number; ci95: [number, number] };
export type Metrics = { R2: Interval; MAE: Interval; "Capture@20%": Interval };
export type Segment = {
  segment: string; customers: number; mean_future_spend: number; bought_again: number;
  "share of customers": number; "share of future money": number;
};
export type Summary = {
  model: string; columns: string[]; cutoff_train: string; cutoff_test: string;
  n_train: number; n_test: number; zero_share_train: number; zero_share_test: number;
  test: { past_spend: Metrics; model_C_headline: Metrics };
  capture_ceiling: number; classifier_auc: number;
  diff_vs_past_spend: Record<string, { point: number; ci: [number, number] }>;
  targeting: { model_C_mean_future_spend: number; past_spend_mean_future_spend: number; average_customer: number; model_C_share_of_money: number };
  high_value_cutoff_gbp: number; segments: Segment[];
  model_comparison: { model: string; R2: number; MAE: number; "Capture@20%": number }[];
  shap: { feature: string; share: number }[];
  raw_rows: number; raw_customers: number; raw_first_date: string; raw_last_date: string;
};
export type Preset = {
  segment: string; customer_id: string;
  features: { Recency: number; Frequency: number; Monetary: number; IPI: number | null };
  predicted_spend: number; probability_buy_again: number; actual_spend: number;
};
export type ScorerData = { presets: Preset[] };
export type Schema = { column: string; type?: string; meaning: string; example?: string; null_pct?: number; p5?: number; median?: number; p95?: number };
export type Dataset = {
  followed_customer: string;
  raw: { schema: Schema[]; columns: string[]; rows: (string | number)[][]; note?: string };
  features: { schema: Schema[] };
};
export type Score = {
  predicted_spend_gbp: number; multiplier: number; probability_buy_again: number; segment: string;
  reasons?: unknown;
};

export type FlowStage = {
  id: string; title: string; did: string; why: string; found: string; source: string; table?: "models" | "features" | "targeting"; aside?: { title: string; code: string; text: string; example_title?: string; example?: string[] };
};
export type Decision = {
  id: string; title: string; evidence: string; how: string; result: string; limit: string;
};
export type Story = {
  flow: FlowStage[]; features_table: string[][]; decisions: Decision[]; not_done: string[];
};

export type HeuristicRow = {
  name: string; how: string; fitted_on_train: string; R2: number; MAE: number; "Capture@20%": number; total_pred_over_actual: number;
};
type Period = { customers: number; arpu_active: number; churn: number; mean_all: number; clv_formula1: number };
export type Heuristics = {
  textbook: { arpu: number; margin: number; churn: number; clv: number };
  ours: { train: Period; test: Period; lifetimes_periods_train: number };
  fitted_k: number; implied_lifespan_fraction: number; table: HeuristicRow[];
  diagnostic: { test_best_k: number; model_C_rescaled_to_actual_total: { R2: number; MAE: number }; past_spend_ratio_train: number; past_spend_ratio_test: number };
};

export type TrainExamples = {
  cutoff: string; n_train: number;
  rows: { customer_id: string; Recency: number; Frequency: number; Monetary: number; IPI: number | null; OverdueRatio: number | null; target: number; label_ratio: number }[];
};

export type TargetingRules = {
  n_test: number; n_top: number; average_customer: number;
  rows: { rule: string; ranks_by: string; mean_future_spend: number; lift_vs_average: number; share_of_money: number }[];
};
