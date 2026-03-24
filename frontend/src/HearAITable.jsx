import { useMemo, useState } from "react";
import { cols, data } from "./faultData";

export default function HearAITable() {
  const [active, setActive] = useState("all");
  const sections = useMemo(
    () => (active === "all" ? Object.values(data) : [data[active]]),
    [active]
  );

  return (
    <div className="table-wrap">
      <div className="tabs">
        <button className={active === "all" ? "tab active" : "tab"} onClick={() => setActive("all")}>All</button>
        <button className={active === "healthy" ? "tab active" : "tab"} onClick={() => setActive("healthy")}>Healthy</button>
        <button className={active === "bearing" ? "tab active" : "tab"} onClick={() => setActive("bearing")}>Bearing</button>
        <button className={active === "propeller" ? "tab active" : "tab"} onClick={() => setActive("propeller")}>Propeller</button>
      </div>
      {sections.map((section) => (
        <div key={section.label} className="section-block">
          <h3 style={{ color: section.color }} className="section-title">{section.label}</h3>
          <table>
            <thead>
              <tr>
                {cols.map((c) => (
                  <th key={c.key}>{c.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {section.rows.map((row) => (
                <tr key={row.class}>
                  {cols.map((c) => (
                    <td key={c.key}>{row[c.key]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
