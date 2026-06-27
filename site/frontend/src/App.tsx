import { useEffect, useMemo, useState } from "react";
import { Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Download, Info, Menu, RotateCcw, X } from "lucide-react";
import { api, type FilterOptions, type Filters, type Forecast, type Granularity, type Overview, type SegmentRow, type TrendPoint } from "./api";

const defaultFilters: Filters = {
  source_dataset: "",
  region: "",
  occupation_group: "",
  period: "",
};

const currency = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
const percent = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });

const occupationLabels: Record<string, string> = {
  HoReCa: "Гостинично-ресторанный бизнес",
  unknown: "Не определено",
};

function formatSalary(value: number | null | undefined) {
  return value == null ? "-" : `${currency.format(value)} ₽`;
}

function formatPercent(value: number | null | undefined) {
  return value == null ? "-" : `${percent.format(value * 100)}%`;
}

function formatPeriod(value: string | null | undefined, granularity: Granularity) {
  if (!value) return "Последний доступный";
  if (granularity === "quarterly") {
    const match = value.match(/^(\d{4})Q([1-4])$/);
    return match ? `${match[2]} кв. ${match[1]}` : value;
  }
  const match = value.match(/^(\d{4})-(\d{2})$/);
  if (!match) return value;
  return new Intl.DateTimeFormat("ru-RU", { month: "short", year: "numeric" })
    .format(new Date(Number(match[1]), Number(match[2]) - 1, 1))
    .replace(" г.", "");
}

function displayOccupation(value: string) {
  return occupationLabels[value] ?? value;
}

function SelectField({ label, name, value, values, onChange, allLabel, formatValue = (item) => item }: { label: string; name: keyof Filters; value: string; values: string[]; onChange: (value: string) => void; allLabel: string; formatValue?: (item: string) => string }) {
  return <label className="filter-field">
    <span>{label}</span>
    <select data-filter={name} value={value} onChange={(event) => onChange(event.target.value)}>
      <option value="">{allLabel}</option>
      {values.map((item) => <option key={item} value={item}>{formatValue(item)}</option>)}
    </select>
  </label>;
}

export default function App() {
  const [granularity, setGranularity] = useState<Granularity>("quarterly");
  const [draft, setDraft] = useState<Filters>(defaultFilters);
  const [filters, setFilters] = useState<Filters>(defaultFilters);
  const [options, setOptions] = useState<FilterOptions | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [rows, setRows] = useState<SegmentRow[]>([]);
  const [history, setHistory] = useState<SegmentRow[]>([]);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [methodologyOpen, setMethodologyOpen] = useState(false);
  const [forecastHelpOpen, setForecastHelpOpen] = useState(false);

  useEffect(() => {
    api.filters(granularity, draft).then((nextOptions) => {
      setOptions(nextOptions);
      setDraft((current) => ({
        ...current,
        period: nextOptions.periods.includes(current.period) ? current.period : "",
      }));
    }).catch((reason: Error) => setError(reason.message));
  }, [granularity, draft.region, draft.occupation_group]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    const selected = { ...filters };
    const hasExactSegment = Boolean(selected.region && selected.occupation_group);
    Promise.all([
      api.overview(granularity, selected),
      api.trend(granularity, selected),
      api.segments(granularity, selected),
      hasExactSegment ? api.history(granularity, selected) : Promise.resolve([]),
      hasExactSegment ? api.forecast(granularity, selected) : Promise.resolve(null),
    ]).then(([nextOverview, nextTrend, nextRows, nextHistory, nextForecast]) => {
      if (!active) return;
      setOverview(nextOverview);
      setTrend(nextTrend);
      setRows(nextRows);
      setHistory(nextHistory);
      setForecast(nextForecast);
    }).catch((reason: Error) => {
      if (active) setError(reason.message);
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [filters, granularity]);

  const displayedHistory = useMemo(() => {
    const withDeltas = history.map((row, index) => {
      const previous = history[index - 1];
      const salaryDelta = previous?.median_salary_mid && row.median_salary_mid
        ? row.median_salary_mid / previous.median_salary_mid - 1
        : null;
      return { ...row, salaryDelta };
    });
    return withDeltas.reverse().slice(0, granularity === "monthly" ? 18 : 16);
  }, [history, granularity]);

  function applyFilters() {
    const selected = { ...draft, source_dataset: "" };
    document.querySelectorAll<HTMLSelectElement>("select[data-filter]").forEach((select) => {
      const key = select.dataset.filter as keyof Filters | undefined;
      if (key) selected[key] = select.value;
    });
    setDraft({ ...selected });
    setFilters({ ...selected });
    setSidebarOpen(false);
  }

  function resetFilters() {
    setDraft({ ...defaultFilters });
    setFilters({ ...defaultFilters });
  }

  function changeGranularity(value: Granularity) {
    setGranularity(value);
    setDraft((current) => ({ ...current, period: "" }));
    setFilters((current) => ({ ...current, period: "" }));
  }

  function exportHistory() {
    const context = [
      ["Регион", filters.region || "Все регионы"],
      ["Профессиональная группа", displayOccupation(filters.occupation_group) || "Все группы"],
      ["Режим", granularity === "monthly" ? "Месяц" : "Квартал"],
      ["Выгружено", new Date().toLocaleString("ru-RU")],
      [],
    ];
    const header = ["Период", "Медианная зарплата, ₽", "Диапазон p25-p75, ₽", "Вакансии"];
    const records = history.map((row) => [formatPeriod(row.period, granularity), row.median_salary_mid ?? "", `${row.p25_salary_mid ?? ""}-${row.p75_salary_mid ?? ""}`, row.vacancy_count]);
    const csv = [...context, header, ...records].map((record) => record.join(";")).join("\n");
    const href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = href;
    link.download = `labor-market-${granularity}.csv`;
    link.click();
    URL.revokeObjectURL(href);
  }

  const titlePeriod = formatPeriod(overview?.period ?? filters.period, granularity);
  const granularityLabel = granularity === "monthly" ? "месячный" : "квартальный";
  const forecastDelta = forecast ? forecast.forecast_salary - forecast.baseline_salary : null;
  const forecastPeriod = forecast ? formatPeriod(forecast.target_period, granularity) : "следующий период";
  const vacancyPeriodLabel = filters.period ? `за ${titlePeriod}` : "в последнем доступном периоде";

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><button className="icon-button mobile-only" onClick={() => setSidebarOpen(true)} aria-label="Открыть фильтры"><Menu size={20} /></button><strong>Прогноз зарплат</strong></div>
      <button className="export-button" onClick={exportHistory} title="Скачать историю выбранного сегмента в CSV"><Download size={16} /> CSV</button>
    </header>

    <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
      <div className="sidebar-heading"><strong>Параметры анализа</strong><button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)} aria-label="Закрыть фильтры"><X size={20} /></button></div>
      <div className="segmented" aria-label="Горизонт агрегации">
        <button className={granularity === "monthly" ? "selected" : ""} onClick={() => changeGranularity("monthly")}>Месяц</button>
        <button className={granularity === "quarterly" ? "selected" : ""} onClick={() => changeGranularity("quarterly")}>Квартал</button>
      </div>
      <SelectField label="Регион" name="region" value={draft.region} values={options?.regions ?? []} onChange={(value) => setDraft({ ...draft, region: value, period: "" })} allLabel="Все регионы" />
      <SelectField label="Профессиональная группа" name="occupation_group" value={draft.occupation_group} values={options?.occupation_groups ?? []} onChange={(value) => setDraft({ ...draft, occupation_group: value, period: "" })} allLabel="Все группы" formatValue={displayOccupation} />
      <div className="filter-actions"><button className="primary-button" onClick={applyFilters}>Применить</button><button className="secondary-button" onClick={resetFilters}><RotateCcw size={15} /> Сбросить</button></div>
    </aside>

    <main className="dashboard">
      <section className="page-heading"><p className="eyebrow">{granularity === "monthly" ? "Месячная" : "Квартальная"} аналитика</p><h1>Аналитика рынка труда</h1><p>{filters.region || "Все регионы"} · {filters.occupation_group ? displayOccupation(filters.occupation_group) : "Все группы"} · {titlePeriod}</p></section>
      {error && <div className="notice error">{error}</div>}
      {loading && <div className="notice">Загружаем данные сегмента...</div>}
      <section className="kpi-grid">
        <article><span>Медианная зарплата</span><strong>{formatSalary(overview?.median_salary_mid)}</strong><small>{titlePeriod}</small></article>
        <article><span>Вакансии</span><strong>{overview ? currency.format(overview.vacancy_count) : "-"}</strong><small>{vacancyPeriodLabel}</small></article>
        <article><span>С указанной зарплатой</span><strong>{formatPercent(overview?.salary_coverage)}</strong><small>доля вакансий</small></article>
        <article><span>Прогноз на {forecastPeriod}</span><strong className={forecastDelta != null && forecastDelta < 0 ? "negative" : "positive"}>{forecastDelta == null ? "-" : `${forecastDelta >= 0 ? "+" : ""}${formatSalary(forecastDelta)}`}</strong><small>изменение к {formatPeriod(forecast?.input_period, granularity)}</small></article>
      </section>

      <section className="main-grid">
        <article className="chart-panel">
          <div className="panel-heading"><div><h2>Динамика зарплаты и вакансий</h2><p>Медианная зарплата и число вакансий по периодам</p></div><span className="legend"><i /> Зарплата <b /> Вакансии</span></div>
          <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={trend} margin={{ top: 18, right: 12, left: 0, bottom: 0 }}><CartesianGrid stroke="#2b3237" vertical={false} /><XAxis dataKey="period" tickFormatter={(value) => formatPeriod(value, granularity)} tick={{ fill: "#9ba5aa", fontSize: 11 }} axisLine={{ stroke: "#394146" }} tickLine={false} /><YAxis yAxisId="salary" tickFormatter={(value) => `${Math.round(Number(value) / 1000)}\u00a0тыс.\u00a0₽`} tick={{ fill: "#9ba5aa", fontSize: 11 }} axisLine={false} tickLine={false} width={98} /><YAxis yAxisId="vacancy" orientation="right" tickFormatter={(value) => value >= 1000 ? `${percent.format(value / 1000)}\u00a0тыс.` : currency.format(value)} tick={{ fill: "#9ba5aa", fontSize: 11 }} axisLine={false} tickLine={false} width={58} /><Tooltip contentStyle={{ background: "#151a1e", border: "1px solid #394146", borderRadius: 6 }} labelStyle={{ color: "#f2f5f6" }} labelFormatter={(value) => formatPeriod(String(value), granularity)} formatter={(value, name) => [name === "Зарплата" ? formatSalary(Number(value)) : currency.format(Number(value)), name]} /><Bar yAxisId="vacancy" dataKey="vacancy_count" name="Вакансии" fill="#5b6368" fillOpacity={0.72} radius={[2, 2, 0, 0]} maxBarSize={28} /><Line yAxisId="salary" type="monotone" dataKey="median_salary_mid" name="Зарплата" stroke="#57d4cc" strokeWidth={2} dot={{ r: 3, fill: "#151a1e", strokeWidth: 2 }} activeDot={{ r: 5 }} /></ComposedChart></ResponsiveContainer></div>
        </article>
        <aside className="forecast-panel"><div className="panel-heading"><h2>Прогноз на {forecastPeriod}</h2><button className="icon-button" onClick={() => setForecastHelpOpen((open) => !open)} aria-label="Как рассчитан прогноз" aria-expanded={forecastHelpOpen} title="Как рассчитан прогноз"><Info size={16} /></button></div>{forecast ? <><p className="muted">Текущая медианная зарплата</p><strong className="baseline">{formatSalary(forecast.baseline_salary)}</strong><div className="rule" /><p className="muted">Прогноз медианной зарплаты</p><strong className="forecast-value">{formatSalary(forecast.forecast_salary)}</strong><p className={forecast.predicted_change_percent >= 0 ? "change positive" : "change negative"}>{forecast.predicted_change_percent >= 0 ? "+" : ""}{percent.format(forecast.predicted_change_percent)}% <span>({forecastDelta != null && `${forecastDelta >= 0 ? "+" : ""}${formatSalary(forecastDelta)}`})</span></p>{forecast.q4_to_q1_correction_applied && <div className="correction">Учтена сезонность перехода из IV в I квартал</div>}{forecastHelpOpen && <div className="forecast-help" role="status"><strong>Как рассчитан прогноз</strong><p>Модель CatBoost использует историю выбранного региона и профессиональной группы. Итог показывает ожидаемую медианную зарплату на следующий {granularity === "monthly" ? "месяц" : "квартал"}.</p></div>}<div className="forecast-model"><span>Модель</span><strong>CatBoost</strong></div></> : <p className="muted">Выберите регион и профессиональную группу для расчёта.</p>}</aside>
      </section>

      <section className="table-panel"><div className="panel-heading"><div><h2>Детализация по периодам</h2><p>История выбранного сегмента</p></div><span>{granularityLabel} режим</span></div><div className="table-scroll"><table><thead><tr><th>Период</th><th>Медианная зарплата</th><th>Изменение зарплаты</th><th>Диапазон p25-p75</th><th>Вакансии</th></tr></thead><tbody>{displayedHistory.map((row) => <tr key={row.period}><td>{formatPeriod(row.period, granularity)}</td><td>{formatSalary(row.median_salary_mid)}</td><td className={row.salaryDelta != null && row.salaryDelta < 0 ? "negative" : "positive"}>{formatPercent(row.salaryDelta)}</td><td>{formatSalary(row.p25_salary_mid)} - {formatSalary(row.p75_salary_mid)}</td><td>{currency.format(row.vacancy_count)}</td></tr>)}{!loading && displayedHistory.length === 0 && <tr><td colSpan={5} className="empty-cell">Для выбранного сегмента нет доступной истории.</td></tr>}</tbody></table></div></section>
      </main>

      <footer className="app-footer"><span>Прогнозы строятся по историческим данным вакансий.</span><button onClick={() => setMethodologyOpen(true)}>Методология</button></footer>

    {methodologyOpen && <div className="modal-backdrop" role="presentation" onMouseDown={() => setMethodologyOpen(false)}><section className="methodology-modal" role="dialog" aria-modal="true" aria-label="Методология" onMouseDown={(event) => event.stopPropagation()}><button className="icon-button modal-close" onClick={() => setMethodologyOpen(false)} aria-label="Закрыть"><X size={18} /></button><h2>Методология</h2><p>Зарплата агрегируется по региону, профессиональной группе и выбранному периоду. Для отображения используется медианная зарплата, чтобы отдельные экстремальные значения меньше влияли на результат.</p><p>Прогноз рассчитывается моделью CatBoost по историческим данным выбранного сегмента и показывает ожидаемое значение следующего периода.</p></section></div>}
  </div>;
}
