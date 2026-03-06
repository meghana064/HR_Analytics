# AI Workforce Guardian — Power BI Dashboard Blueprint

**Build this dashboard in ~20 minutes.** Follow each step in order.

---

## Part 1: Import Data (2 min)

1. Open **Power BI Desktop**.
2. **Home** → **Get data** → **Text/CSV**.
3. Browse to:  
   `Hr_analytics\employee_attrition_ai\data\WA_Fn-UseC_-HR-Employee-Attrition.csv`
4. Click **Transform Data** (so we can rename the table).
5. In Power Query Editor: Right-panel → **Rename** your query to `HR_Data`.
6. **Home** → **Close & Apply**.

---

## Part 2: Create Measures (5 min)

1. Click anywhere on the canvas (empty report area).
2. **Modeling** tab → **New measure**.
3. Create each measure below. After pasting, press **Enter** or click ✓.

| # | Measure Name | DAX Formula |
|---|--------------|-------------|
| 1 | Total Employees | `Total Employees = COUNTROWS(HR_Data)` |
| 2 | Attrition Count | `Attrition Count = CALCULATE(COUNTROWS(HR_Data), HR_Data[Attrition] = "Yes")` |
| 3 | Attrition Rate % | `Attrition Rate % = DIVIDE([Attrition Count], [Total Employees], 0) * 100` |
| 4 | Avg Monthly Income | `Avg Monthly Income = AVERAGE(HR_Data[MonthlyIncome])` |
| 5 | Avg Years at Company | `Avg Years at Company = AVERAGE(HR_Data[YearsAtCompany])` |
| 6 | Employee Count | `Employee Count = COUNTROWS(HR_Data)` |

**If your table is NOT named HR_Data:** Replace `HR_Data` with your table name in every formula above.

**For charts:** Use **Employee Count** (or **Attrition Count**) in Values. For "Attrition by Department" stacked bar, use **Employee Count** in Values and **Attrition** in Legend.

---

## Part 3: Page 1 — Company Dashboard (5 min)

1. Rename Page 1: Right-click tab at bottom → **Rename** → `Company Dashboard`.
2. **View** → **Themes** → **Executive** (or **Innovation**) for a dark look.

### Row 1: KPI Cards

1. **Insert** → **Card** (or find Card in Visualizations).
2. Drag **Total Employees** measure into **Fields**.
3. Resize and duplicate: Copy card (Ctrl+C, Ctrl+V) 3 times.
4. Replace field in each copy with:
   - Card 2: **Attrition Rate %**
   - Card 3: **Avg Monthly Income**
   - Card 4: **Avg Years at Company**
5. Format Card 2: Select it → **Format** (paintbrush) → **Value** → Display units: **None** → add suffix **%** in format string if needed.
6. Format Card 3: **Value** → Format: **Currency** → **$ English**.

### Row 2: Attrition by Department

1. **Insert** → **Stacked bar chart**.
2. **Axis:** Department  
   **Legend:** Attrition  
   **Values:** Employee Number (Count) — or create measure `Employee Count = COUNTROWS(HR_Data)` and use that.
3. Resize as needed.

### Row 3: Two charts side by side

1. **Clustered column chart**  
   - **Axis:** JobRole  
   - **Values:** Attrition Count  
   - Sort by Attrition Count descending (click ⋮ on chart → Sort)

2. **Clustered column chart**  
   - **Axis:** OverTime  
   - **Legend:** Attrition  
   - **Values:** Employee Number (Count)

### Row 4: Age Distribution

1. **Insert** → **Clustered column chart** or **Histogram** (if available).
2. **Axis:** Age  
3. **Values:** Employee Number (Count)
4. If Age has too many values: In **Data** pane, select Age → **Modeling** → **Summarization** → **Count (Distinct)** or group Age into bins (Add Column in Power Query first).

**Quick Age bins:** In Power Query, add column: `Age Group = Number.RoundDown([Age]/10)*10 & "-" & Number.RoundDown([Age]/10)*10+9` → use Age Group on axis instead.

---

## Part 4: Page 2 — HR Insights (4 min)

1. **Insert** → **New page** (or right-click page tab → Duplicate).
2. Rename to `HR Insights`.

### Slicer + Cards

1. **Insert** → **Slicer**.
2. Add **Department** to Slicer.
3. Style: **List** or **Dropdown** (Format → Slicer → Options).

4. Add 4 Cards: Total Employees, Attrition Rate %, Avg Monthly Income, Avg Years at Company.  
   (These will filter by Department automatically.)

### Charts

1. **Clustered bar chart:**  
   - **Axis:** MonthlyIncome  
   - **Legend:** Attrition  
   Or: **Box plot** if your Power BI version has it (MonthlyIncome by Attrition).

2. **Bar chart:**  
   - **Axis:** YearsAtCompany  
   - **Values:** Attrition Count

### Table

1. **Insert** → **Table**.
2. Add: EmployeeNumber, Department, JobRole, MonthlyIncome, YearsAtCompany, Attrition.
3. Optional: **Filter** pane → Add filter: Attrition = "Yes" to show only those who left.

---

## Part 5: Page 3 — Attrition Data (2 min)

1. **Insert** → **New page** → Rename `Attrition Data`.
2. **Insert** → **Table**.
3. Add all columns (or: EmployeeNumber, Department, JobRole, MonthlyIncome, Attrition, etc.).
4. **Filter** pane → Add filter for this visual: **Attrition** = **Yes**.
5. Add **Department** slicer to drill down.

---

## Part 6: Sync Slicers (1 min)

1. **View** → **Sync slicers**.
2. Drag **Department** slicer from one page into the **Sync** panel.
3. Check the pages where it should apply (e.g., Company Dashboard, HR Insights, Attrition Data).

---

## Part 7: Apply Dark Theme (Optional)

**Option A — Use project theme file:**
1. **View** → **Themes** → **Browse for themes**.
2. Select `powerbi/theme_workforce_guardian.json` from this project.
3. Theme applies dark background (#0f172a) and accent (#e50914).

**Option B — Built-in theme:**
1. **View** → **Themes** → **Executive** or **Innovation** (dark themes).

---

## Layout Summary

```
┌─────────────────────────────────────────────────────────────┐
│  COMPANY DASHBOARD                                           │
├─────────┬─────────┬──────────────┬──────────────────────────┤
│ Total   │Attrition│ Avg Monthly   │ Avg Years               │
│Employees│ Rate %  │ Income       │ at Company               │
├─────────┴─────────┴──────────────┴──────────────────────────┤
│  [ Attrition by Department - Stacked Bar ]                   │
├────────────────────────────┬────────────────────────────────┤
│ [ Attrition by Job Role ]  │ [ Overtime vs Attrition ]       │
├────────────────────────────┴────────────────────────────────┤
│  [ Age Distribution ]                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  HR INSIGHTS                                                │
├──────────┬──────────────────────────────────────────────────┤
│Department│  [Cards: Total, Attrition %, Income, Years]       │
│ Slicer   ├──────────────────────────────────────────────────┤
│          │  [ MonthlyIncome by Attrition ] [ Years Chart ]   │
│          ├──────────────────────────────────────────────────┤
│          │  [ Employee Table ]                               │
└──────────┴──────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ATTRITION DATA  (Filter: Attrition = Yes)                   │
│  [ Department Slicer ]                                        │
│  [ Full Employee Table ]                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Save Your Report

**File** → **Save as** → `AI_Workforce_Guardian.pbix`

---

**Done.** Your Power BI dashboard matches the Streamlit app's analytics views.
