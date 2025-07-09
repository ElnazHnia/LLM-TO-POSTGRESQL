from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re

class GridPos(BaseModel):
    x: int
    y: int
    w: int
    h: int

class Datasource(BaseModel):
    type: str = "postgres"
    uid: str = "aeo8prusu1i4gc"

class Target(BaseModel):
    refId: str = "A"
    format: str  # "time_series" or "table"
    rawSql: str

class Tooltip(BaseModel):
    mode: str = "single"

class Legend(BaseModel):
    displayMode: str = "list"
    placement: str = "bottom"

class ReduceOptions(BaseModel):
    calcs: List[str] = ["lastNotNull"]
    fields: str = ""
    values: bool = True

class PanelOptions(BaseModel):
    tooltip: Tooltip = Tooltip()
    legend: Legend = Legend()
    reduceOptions: ReduceOptions = ReduceOptions()

class Panel(BaseModel):
    id: int
    type: str  # "barchart", "piechart", "table", "timeseries", "stat"
    title: str
    datasource: Datasource = Datasource()
    targets: List[Target]
    gridPos: GridPos
    options: PanelOptions = PanelOptions()

class TimeRange(BaseModel):
    from_time: str = Field(alias="from", default="now-5y")
    to: str = "now"

class TimePicker(BaseModel):
    refresh_intervals: List[str] = ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
    time_options: List[str] = ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d", "90d", "6M", "1y", "5y"]

class DashboardContent(BaseModel):
    title: str
    schemaVersion: int = 36
    version: int = 1
    refresh: str = "5s"
    time: TimeRange = TimeRange()
    timepicker: TimePicker = TimePicker()
    panels: List[Panel]

class DashboardSpec(BaseModel):
    dashboard: DashboardContent
    overwrite: bool = False

    class Config:
        allow_population_by_field_name = True

class Target(BaseModel):
    refId:  str = Field(
        "A",
        description="Reference ID for this query/target"
    )
    format: str = Field(
        ...,
        description="Grafana data format: 'time_series' or 'table'"
    )
    rawSql: str = Field(
        ...,
        description="PostgreSQL query executed for this target"
    )

class DashboardValidation:
    
    @staticmethod
    def validate_dashboard_dict(
        dashboard_dict: dict,
        prompt: str,
        timestamp: str,
        *,
        table_meta: Dict[str, Dict[str, Any]] | None = None  # NEW
    ) -> tuple[bool, str]:
        """
        Validate dashboard dictionary directly using Python logic instead of LLM.
        Returns (is_valid, error_message)
        """
        
        errors = []
        table_meta = table_meta or {}                         # ✱ CHANGED ⇢ default {}
        allowed_tables = {t.upper() for t in table_meta} or {'SALES', 'PERSON', 'PRODUCTS'}  # ✱ CHANGED
        known_columns = {
            col.split()[0].upper()                           # strip "(type)"
            for meta in table_meta.values()
            for col in meta.get("columns", [])
        }                                                    # ✱ CHANGED
        
        try:
            # Extract dashboard data
            dashboard_data = dashboard_dict.get("dashboard", {})
            panels = dashboard_data.get("panels", [])
            title = dashboard_data.get("title", "")
            
            # 1. Title validation
            expected_title_pattern = f".*- {re.escape(timestamp)}"
            if not re.search(expected_title_pattern, title):
                errors.append(f"ERROR: Title '{title}' does not match expected format '[Description] - {timestamp}'")
            
            # 2. Panel type ⇄ format mapping validation
            valid_type_format_mapping = {
                "barchart": "table",
                "piechart": "table", 
                "table": "table",
                "stat": "table",
                "timeseries": "time_series"
            }
            
            for i, panel in enumerate(panels):
                panel_type = panel.get("type", "")
                targets = panel.get("targets", [])
                
                if targets:
                    panel_format = targets[0].get("format", "")
                    expected_format = valid_type_format_mapping.get(panel_type)
                    
                    if expected_format and panel_format != expected_format:
                        errors.append(f"ERROR: Panel {i} ({panel_type}) has format = {panel_format}, expected {expected_format}")
            
            # 3. SQL validation for each panel
            for i, panel in enumerate(panels):
                targets = panel.get("targets", [])
                panel_type = panel.get("type", "")
                
                if targets:
                    raw_sql = targets[0].get("rawSql", "")
                    if raw_sql:
                        sql_errors = DashboardValidation.validate_sql_query(
                            raw_sql,
                            i,
                            panel_type,
                            allowed_tables=allowed_tables,      # ✱ CHANGED
                            known_columns=known_columns         # ✱ CHANGED
                        )
                        errors.extend(sql_errors)
            
            # 4. Grid position validation
            grid_errors = DashboardValidation.validate_grid_positions(panels)
            errors.extend(grid_errors)
            
            if errors:
                return False, "\n".join(errors)
            else:
                return True, ""
                
        except Exception as e:
            return False, f"ERROR: Error during validation: {str(e)}"

    @staticmethod
    def validate_sql_query(
        sql: str,
        panel_index: int,
        panel_type: str,
        *,
        allowed_tables: set,
        known_columns: set
    ) -> List[str]:
        """
        Validate a single SQL query and return list of errors.
        """
        errors = []
        
               
        # 2. Check for allowed tables (must be plural)
        forbidden_singular = {'SALE', 'PRODUCT'}
        
        # Extract table names from FROM and JOIN clauses
        from_tables = DashboardValidation.extract_table_names_from_sql(sql)
        from_tables = {t for t in from_tables if t.upper() not in known_columns}
        for table in from_tables:
            if table.upper() in forbidden_singular:
                errors.append(f"ERROR: Table '{table}' should be plural")
            elif table.upper() not in allowed_tables:       # ✱ CHANGED
                errors.append(f"ERROR: Table '{table}' not in allowed tables: {', '.join(sorted(allowed_tables))}")
        
        # 3. Check alias usage vs declarations
        alias_errors = DashboardValidation.validate_sql_aliases(sql, panel_index)
        errors.extend(alias_errors)
        
        # 4. Check for ::text casting in bar/pie charts
        if panel_type in ['barchart', 'piechart']:
            text_cast_errors = DashboardValidation.validate_text_casting(sql, panel_index, panel_type)
            errors.extend(text_cast_errors)
        
        # 5. Check for proper JOINs when referencing other tables
        join_errors = DashboardValidation.validate_required_joins(sql, panel_index)
        errors.extend(join_errors)
        
        # 6. Check for column references that might be mistaken for tables
        column_errors = DashboardValidation.validate_column_references(
            sql,
            panel_index,
            known_columns=known_columns                      # ✱ CHANGED
        )
        errors.extend(column_errors)
        
        return errors

    @staticmethod
    def extract_table_names_from_sql(sql: str) -> set:
        """
        Extract table names from FROM and JOIN clauses - IMPROVED VERSION
        """
        tables = set()
        sql_upper = sql.upper()
        
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?'
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?'
        
        tables.update(re.findall(from_pattern, sql_upper))
        tables.update(re.findall(join_pattern, sql_upper))
        
        return tables

   
    @staticmethod
    def validate_column_references(sql: str, panel_index: int) -> List[str]:
        errors = []
        sql_upper = sql.upper()
        for col in ['SALE_DATE', 'PRODUCT_ID', 'PERSON_ID']:
            # only match “FROM <col>” or “JOIN <col>” when NOT inside a function call
            pattern = rf'(?<!\()\b(?:FROM|JOIN)\s+{col}\b'
            if re.search(pattern, sql_upper):
                errors.append(
                    f"ERROR: '{col}' appears to be used as a table name, but it's a column (Panel {panel_index})"
                )
        return errors

    
    @staticmethod
    def validate_sql_aliases(sql: str, panel_index: int) -> List[str]:
        """
        Validate that all aliases used in SELECT/WHERE/GROUP/ORDER are declared in FROM/JOIN
        """
        errors = []
        
        # Extract declared aliases/tables from FROM/JOIN
        declared_aliases = DashboardValidation.extract_declared_aliases(sql)
        
        # Extract used aliases from SELECT/WHERE/GROUP/ORDER
        used_aliases = DashboardValidation.extract_used_aliases(sql)
        
        # Check if all used aliases are declared
        undeclared = used_aliases - declared_aliases
        if undeclared:
            for alias in undeclared:
                errors.append(f"ERROR: Alias '{alias}' not declared in FROM/JOIN (Panel {panel_index})")
                
                # Provide specific guidance for common issues
                if alias.upper() == 'PERSON':
                    errors.append(f"ERROR: Panel {panel_index} references person table but doesn't JOIN it")
                elif alias.upper() == 'PRODUCTS':
                    errors.append(f"ERROR: Panel {panel_index} references products table but doesn't JOIN it")
        
        return errors

    @staticmethod
    def extract_declared_aliases(sql: str) -> set:
        """
        Extract all table names and aliases from FROM/JOIN clauses
        """
        declared = set()
        sql_upper = sql.upper()
        
        # Pattern to match FROM/JOIN table_name [AS] alias or just table_name
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?'
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?'
        
        from_matches = re.findall(from_pattern, sql_upper)
        join_matches = re.findall(join_pattern, sql_upper)
        
        for table, alias in from_matches + join_matches:
            declared.add(table)  # Add the table name itself
            if alias:  # Add the alias if it exists
                declared.add(alias)
        
        return declared

    @staticmethod
    def extract_used_aliases(sql: str) -> set:
        """
        Extract all prefixes used before dots in SELECT/WHERE/GROUP/ORDER clauses
        """
        used = set()
        sql_upper = sql.upper()
        
        # Pattern to match table_name.column_name or alias.column_name
        # Only look in SELECT, WHERE, GROUP BY, ORDER BY, and HAVING clauses
        relevant_parts = []
        
        # Extract different parts of the query
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if select_match:
            relevant_parts.append(select_match.group(1))
        
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)', sql_upper, re.DOTALL)
        if where_match:
            relevant_parts.append(where_match.group(1))
        
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|\s+HAVING|$)', sql_upper, re.DOTALL)
        if group_match:
            relevant_parts.append(group_match.group(1))
        
        order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', sql_upper, re.DOTALL)
        if order_match:
            relevant_parts.append(order_match.group(1))
        
        having_match = re.search(r'HAVING\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_upper, re.DOTALL)
        if having_match:
            relevant_parts.append(having_match.group(1))
        
        # Find alias.column patterns in relevant parts
        alias_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*'
        
        for part in relevant_parts:
            matches = re.findall(alias_pattern, part)
            used.update(matches)
        
        return used

    @staticmethod
    def validate_text_casting(sql: str, panel_index: int, panel_type: str) -> List[str]:
        """
        Validate that numeric fields used for grouping in bar/pie charts end with ::text.
        """
        errors = []
        
        if panel_type in ['barchart', 'piechart']:
            # Look for EXTRACT functions that should be cast to text
            extract_pattern = r'EXTRACT\s*\(\s*(?:YEAR|MONTH|DAY|HOUR)\s+FROM\s+[^)]+\)'
            
            matches = re.findall(extract_pattern, sql.upper())
            if matches and '::TEXT' not in sql.upper():
                errors.append(f"ERROR: Panel {panel_index} ({panel_type}) should cast EXTRACT result to ::text")
        
        return errors

    @staticmethod
    def validate_required_joins(sql: str, panel_index: int) -> List[str]:
        """
        Validate that required JOINs are present when referencing other tables.
        """
        errors = []
        sql_upper = sql.upper()
        
        # Get all declared tables
        declared_tables = {t.upper() for t in DashboardValidation.extract_table_names_from_sql(sql)}
        
        # Check if person.something is used but person table is not in FROM/JOIN
        if 'PERSON.' in sql_upper and 'PERSON' not in declared_tables:
            errors.append(f"ERROR: Panel {panel_index} references person table but doesn't JOIN it")
        
        # Check if products.something is used but products table is not in FROM/JOIN  
        if 'PRODUCTS.' in sql_upper and 'PRODUCTS' not in declared_tables:
            errors.append(f"ERROR: Panel {panel_index} references products table but doesn't JOIN it")
        
        return errors

    @staticmethod
    def validate_grid_positions(panels: List[dict]) -> List[str]:
        """
        Validate grid positions follow the 2-per-row rule with correct mathematical formula
        """
        errors = []
        total_panels = len(panels)
        
        for i, panel in enumerate(panels):
            grid_pos = panel.get("gridPos", {})
            actual_x = grid_pos.get("x", 0)
            actual_w = grid_pos.get("w", 12)
            actual_y = grid_pos.get("y", 0)
            
            # Calculate expected positions based on the mathematical formula
            if i == total_panels - 1 and total_panels % 2 == 1:
                # Last panel in odd total - should be full width
                expected_x = 0
                expected_w = 24
                expected_y = (i // 2) * 8
            else:
                # Regular 2-panel layout
                expected_x = 0 if i % 2 == 0 else 12
                expected_w = 12
                expected_y = (i // 2) * 8
            
            # Check if the position matches expected values
            if (actual_x != expected_x or actual_w != expected_w or actual_y != expected_y):
                errors.append(f"ERROR: Panel {i} grid position incorrect. Expected (x={expected_x}, y={expected_y}, w={expected_w}), got (x={actual_x}, y={actual_y}, w={actual_w})")
        
        return errors

    @staticmethod
    def validate_dashboard_python(
        dashboard_dict: dict,
        prompt: str,
        timestamp: str,
        *,
        table_meta: Dict[str, Dict[str, Any]] | None = None   # NEW
    ) -> tuple[bool, str]:
        """
        Main validation function that replaces ask_ollama_if_valid.
        This function validates the dashboard dictionary using pure Python logic.
        """
        
        print("\n── Python Validation ──────────────────────────────")
        
        dashboard_data = dashboard_dict.get("dashboard", {})
        panels = dashboard_data.get("panels", [])
        
        print("SQL Queries Found:")
        for i, panel in enumerate(panels):
            title = panel.get("title", "Untitled")
            targets = panel.get("targets", [])
            if targets:
                sql = targets[0].get("rawSql", "")
                print(f"  [panel {i}] {title}")
                print(f"  SQL: {sql}")
        
        is_valid, error_message = DashboardValidation.validate_dashboard_dict(
            dashboard_dict,
            prompt,
            timestamp,
            table_meta=table_meta                           # ✱ CHANGED
        )
        
        if is_valid:
            print("VALIDATION PASSED")
        else:
            print("Validation failed:")
            print(error_message)
        
        print("────────────────────────────────────────────────")
        
        return is_valid, error_message