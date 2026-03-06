// Power Query (M) - Add Employee Name column
// In Power Query Editor: Add Column → Custom Column
// Name: EmployeeName
// Custom column formula:

"Employee " & Text.From([EmployeeNumber])

// Alternative: Use Index if EmployeeNumber is not sequential
// "Employee " & Text.From([Index])
