export type Granularity = "monthly" | "quarterly";

export type Filters = {
  source_dataset: string;
  region: string;
  occupation_group: string;
  period: string;
};

export type FilterOptions = {
  granularity: Granularity;
  source_datasets: string[];
  regions: string[];
  occupation_groups: string[];
  periods: string[];
};

export type Overview = {
  granularity: Granularity;
  period: string;
  segment_count: number;
  median_salary_mid: number | null;
  vacancy_count: number;
  salary_coverage: number | null;
};

export type TrendPoint = {
  period: string;
  median_salary_mid: number | null;
  vacancy_count: number;
  salary_coverage: number | null;
};

export type SegmentRow = {
  source_dataset: string;
  region: string;
  occupation_group: string;
  period: string;
  median_salary_mid: number | null;
  p25_salary_mid: number | null;
  p75_salary_mid: number | null;
  vacancy_count: number;
  salary_count: number;
  employer_count: number;
};

export type Forecast = {
  granularity: Granularity;
  source_dataset: string;
  region: string;
  occupation_group: string;
  input_period: string;
  target_period: string;
  baseline_salary: number;
  catboost_salary: number;
  forecast_salary: number;
  predicted_change_percent: number;
  q4_to_q1_correction_applied: boolean;
  prediction_strategy: string;
};

const API_URL = "http://127.0.0.1:8000/api";

function query(params: Record<string, string | undefined>) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value) search.set(key, value);
  }
  return search.toString();
}

async function request<T>(path: string, params: Record<string, string | undefined> = {}) {
  const suffix = query(params);
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}${suffix ? `?${suffix}` : ""}`);
  } catch {
    throw new Error("Не удалось загрузить данные. Повторите попытку.");
  }
  if (!response.ok) {
    if (response.status === 404) throw new Error("Для выбранных параметров данных нет. Измените фильтры.");
    throw new Error("Не удалось загрузить данные. Попробуйте изменить фильтры.");
  }
  return (await response.json()) as T;
}

export const api = {
  filters: (granularity: Granularity, filters: Partial<Filters> = {}) => request<FilterOptions>("/filters", { granularity, ...filters }),
  overview: (granularity: Granularity, filters: Filters) => request<Overview>("/overview", { granularity, ...filters }),
  trend: (granularity: Granularity, filters: Filters) => request<TrendPoint[]>("/trend", { granularity, ...filters }),
  segments: (granularity: Granularity, filters: Filters) => request<SegmentRow[]>("/segments", { granularity, ...filters, limit: "100" }),
  history: (granularity: Granularity, filters: Filters) => request<SegmentRow[]>("/segment-history", { granularity, ...filters }),
  forecast: (granularity: Granularity, filters: Filters) => request<Forecast>("/forecast", { granularity, ...filters }),
};
