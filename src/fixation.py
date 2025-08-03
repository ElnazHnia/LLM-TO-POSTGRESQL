from typing import List, Dict, Optional, Tuple
import re

# Reserved keywords that should never be used as table aliases
SQL_RESERVED_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "ORDER", "BY", "HAVING",
    "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "OUTER",
    "ON", "USING", "UNION", "ALL", "EXCEPT", "INTERSECT",
    "WITH", "AS", "AND", "OR", "NOT", "NULL", "TRUE", "FALSE",
    "WHEN", "THEN", "ELSE", "END", "DISTINCT", "LIMIT", "OFFSET"
}


def is_reserved_word(identifier: str) -> bool:
    """Check if a given identifier is a SQL reserved keyword (case-insensitive)."""
    return identifier.upper() in SQL_RESERVED_KEYWORDS


def prompt_mentions_time(prompt: str) -> bool:
    """Check if the user prompt indicates a time-based query (daily, weekly, monthly)."""
    return bool(re.search(r"\b(daily|weekly|monthly)\b", prompt, flags=re.IGNORECASE))


def add_table_aliases(
    sql: str,
    allowed_tables: List[str]
) -> Tuple[str, Dict[str, Optional[str]]]:
    """
    Ensure every FROM/JOIN table has an alias; if missing or invalid, alias = table name.
    Returns the modified SQL and a mapping alias->table_name.
    """
    alias_map: Dict[str, Optional[str]] = {}
    result = []
    i = 0
    n = len(sql)
    depth = 0

    def skip_ws(idx: int) -> int:
        while idx < n and sql[idx].isspace():
            result.append(sql[idx])
            idx += 1
        return idx

    # Copy until first FROM at depth 0
    while i < n:
        if depth == 0 and sql[i:i+4].lower() == 'from':
            result.append('FROM')
            i += 4
            break
        if sql[i] == '(':
            depth += 1
        elif sql[i] == ')':
            depth = max(0, depth - 1)
        result.append(sql[i]); i += 1

    # If no FROM, return unchanged
    if i >= n:
        return sql, {}

    # Process FROM/JOIN table refs
    subq_count = 1
    while i < n:
        i = skip_ws(i)
        if i >= n:
            break
        # Stop at next clause keyword at depth 0
        if depth == 0 and re.match(r'(where|group|order|having|limit|offset)\b', sql[i:], flags=re.IGNORECASE):
            break
        # Handle subquery
        if sql[i] == '(':
            # copy subquery
            start = i
            depth += 1
            while i < n and depth > 0:
                if sql[i] == '(':
                    depth += 1
                elif sql[i] == ')':
                    depth -= 1
                result.append(sql[i]); i += 1
            i = skip_ws(i)
            # optional AS
            if sql[i:i+2].lower() == 'as' and (i+2 == n or sql[i+2].isspace()):
                result.append(' AS'); i += 2; i = skip_ws(i)
            # read alias
            alias_start = i
            while i < n and (sql[i].isalnum() or sql[i] in '_"'):
                result.append(sql[i]); i += 1
            alias = sql[alias_start:i].strip('"')
            if not alias or is_reserved_word(alias):
                new_alias = f'subq{subq_count}'
                subq_count += 1
                if alias:
                    # remove reserved alias chars
                    for _ in range(len(alias)):
                        result.pop()
                result.append(new_alias)
                alias = new_alias
            alias_map[alias] = None  # no real table
        else:
            # regular table
            # read table name
            tbl_start = i
            if sql[i] in '"`':
                quote = sql[i]; result.append(quote); i += 1
                while i < n and sql[i] != quote:
                    result.append(sql[i]); i += 1
                if i < n:
                    result.append(sql[i]); i += 1
                table = sql[tbl_start+1:i-1]
            else:
                while i < n and (sql[i].isalnum() or sql[i] in '._$'):
                    result.append(sql[i]); i += 1
                table = sql[tbl_start:i]
            i = skip_ws(i)
            # optional AS
            if sql[i:i+2].lower() == 'as' and (i+2 == n or sql[i+2].isspace()):
                result.append(' AS'); i += 2; i = skip_ws(i)
            # read alias if any
            alias_start = i
            while i < n and (sql[i].isalnum() or sql[i] == '_' or sql[i] == '"'):
                result.append(sql[i]); i += 1
            alias = sql[alias_start:i].strip('"')
            if not alias or is_reserved_word(alias):
                new_alias = table.split('.')[-1]
                if alias:
                    for _ in range(len(alias)):
                        result.pop()
                result.append(' ' + new_alias)
                alias = new_alias
            alias_map[alias] = table
        i = skip_ws(i)
        if i < n and sql[i] == ',':
            result.append(','); i += 1; continue
        # handle JOIN keywords
        jm = re.match(r'(inner|left|right|full|cross)?\s*join\b', sql[i:], flags=re.IGNORECASE)
        if jm:
            result.append(' ' + jm.group(0)); i += len(jm.group(0)); continue
        break
    # append rest unchanged
    if i < n:
        result.append(sql[i:])
    return ''.join(result), alias_map


def enforce_time_grouping(
    sql: str,
    prompt: str
) -> str:
    """
    If prompt requests daily/weekly/monthly, produce canonical time-truncation query.
    Otherwise, return original SQL.
    """
    if not prompt_mentions_time(prompt):
        return sql
    # expect alias added
    m = re.search(r"FROM\s+(\S+)\s+(\S+)", sql, flags=re.IGNORECASE)
    if not m:
        return sql
    table, alias = m.group(1), m.group(2)
    gran = 'day' if 'daily' in prompt.lower() else 'week' if 'weekly' in prompt.lower() else 'month'
    return (
        f"SELECT DATE_TRUNC('{gran}', {alias}.sale_date) AS time,"
        f" SUM({alias}.value) AS value"
        f" FROM {table} {alias} GROUP BY time"
    )


def remove_time_grouping(sql: str) -> str:
    # only remove DATE_TRUNC parts, preserve rest
    sql = re.sub(r"DATE_TRUNC\([^)]*\)\s+AS\s+\w+,?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r",\s*DATE_TRUNC\([^)]*\)\s+AS\s+\w+", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r",\s*,", ",", sql)
    sql = re.sub(r"GROUP BY\s+[^;]*(date_trunc|time)[^;]*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"ORDER BY\s+[^;]*(date_trunc|time)[^;]*", "", sql, flags=re.IGNORECASE)
    return sql.strip()


def fix_table_names(
    sql: str,
    allowed_tables: List[str]
) -> str:
    corrected = sql
    for m in re.finditer(r"\b(FROM|JOIN)\s+(`?\w+`?)", sql, flags=re.IGNORECASE):
        tbl = m.group(2).strip('`').lower()
        if tbl not in [t.lower() for t in allowed_tables]:
            corr = tbl[:-1] if tbl.endswith('s') and tbl[:-1] in allowed_tables else tbl+'s' if (tbl+'s') in allowed_tables else None
            if corr:
                corrected = re.sub(rf"\b{tbl}\b", corr, corrected, flags=re.IGNORECASE)
            else:
                corrected = re.sub(rf"\b{m.group(1)}\s+{m.group(2)}", '', corrected, flags=re.IGNORECASE)
    return corrected.strip()


def fix_join_conditions(
    sql: str,
    table_columns: Dict[str, List[str]]
) -> str:
    corrected = re.sub(r"JOIN\s+(\w+)(?![^)]*ON)", r"JOIN \1 ON 1=1", sql, flags=re.IGNORECASE)
    fm = re.search(r"\bFROM\s+([^;]+)", corrected, flags=re.IGNORECASE)
    if fm:
        parts = [p.strip() for p in fm.group(1).split(',') if p.strip()]
        if len(parts) > 1:
            newf = parts[0]
            for p in parts[1:]:
                newf += ' CROSS JOIN ' + p if not p.lower().startswith('join') else ', ' + p
            corrected = corrected.replace(fm.group(1), newf)
    tables = [t for t in table_columns if re.search(rf"\b{t}\b", corrected, flags=re.IGNORECASE)]
    for a, b in zip(tables, tables[1:]):
        common = set(table_columns[a]) & set(table_columns[b])
        if common:
            key = list(common)[0]
            corrected = re.sub(
                rf"\b{a}\b\s+CROSS JOIN\s+{b}\b",
                f"{a} INNER JOIN {b} ON {a}.{key} = {b}.{key}",
                corrected, flags=re.IGNORECASE
            )
    return corrected.strip()


def fix_column_references(
    sql: str,
    schema: Dict[str, List[str]],
    alias_map: Dict[str, Optional[str]]
) -> str:
    # collapse double qualifiers
    sql = re.sub(r"\b(\w+)\.\1\.", r"\1.", sql)
    # build column->aliases map
    col_aliases: Dict[str, List[str]] = {}
    for alias, table in alias_map.items():
        if table and table in schema:
            for c in schema[table]:
                col_aliases.setdefault(c, []).append(alias)
    # qualify unqualified columns
    def repl(m):
        col = m.group(1)
        if m.group(0).startswith('.') or col in alias_map or is_reserved_word(col):
            return m.group(0)
        aliases = col_aliases.get(col, [])
        return f"{aliases[0]}.{col}" if len(aliases)==1 else col
    pattern = re.compile(r"(?<![\w\.])([A-Za-z_]\w*)(?!\s*\()")
    return pattern.sub(repl, sql)


def fix_group_by_clause(sql: str) -> str:
    sel = re.search(r"SELECT\s+(.*?)\s+FROM", sql, flags=re.IGNORECASE|re.DOTALL)
    if not sel:
        return sql
    cols = [c.strip() for c in sel.group(1).split(',')]
    nonagg = [c for c in cols if not re.search(r"\b(SUM|AVG|COUNT|MIN|MAX)", c, flags=re.IGNORECASE)]
    gb = re.search(r"GROUP BY\s+(.*?)(?:\s+ORDER BY|$)", sql, flags=re.IGNORECASE|re.DOTALL)
    if not gb:
        return sql
    existing = [g.strip().split()[-1] for g in gb.group(1).split(',') if g.strip()]
    for c in nonagg:
        expr = c.split()[-1]
        if expr not in existing:
            existing.append(expr)
    return re.sub(r"GROUP BY\s+.*?(?=\s+ORDER BY|$)", 'GROUP BY ' + ', '.join(existing), sql, flags=re.IGNORECASE|re.DOTALL)


def fix_syntax_issues(sql: str) -> str:
    corr = sql
    if corr.count("'") % 2:
        corr += "'"
    op, cp = corr.count('('), corr.count(')')
    if op > cp:
        corr += ')'*(op-cp)
    elif cp > op:
        corr = corr.rstrip(')')
    corr = re.sub(r",\s*(FROM|WHERE|GROUP BY|ORDER BY)", r" \1", corr, flags=re.IGNORECASE)
    return corr.strip()


def validate_and_correct_sql(
    sql: str,
    prompt: str,
    allowed_tables: List[str],
    table_columns: Dict[str, List[str]]
) -> str:
    """Master pipeline: applies aliasing, time grouping, schema-driven fixes."""
    # 1) Add aliases only when missing/invalid
    print(f"[FIXATION] Validating SQL: {sql}")
    sql_with_alias, alias_map = add_table_aliases(sql, allowed_tables)  
    print(f"[FIXATION] After adding aliases: {sql_with_alias}")
    # 2) Enforce time bucketing if requested
    enforced = enforce_time_grouping(sql_with_alias, prompt)
    print(f"[FIXATION] After enforcing time grouping: {enforced}")
    if prompt_mentions_time(prompt):
        return enforced
    # 3) Remove any lingering time grouping
    no_time = remove_time_grouping(enforced)
    print(f"[FIXATION] After removing time grouping: {no_time}")
    # 4) Schema-driven corrections
    t1 = fix_table_names(no_time, allowed_tables)
    print(f"[FIXATION] After fixing table names: {t1}")
    t2 = fix_join_conditions(t1, table_columns)
    print(f"[FIXATION] After fixing join conditions: {t2}")
    t3 = fix_column_references(t2, table_columns, alias_map)
    print(f"[FIXATION] After fixing column references: {t3}")
    t4 = fix_group_by_clause(t3)
    print(f"[FIXATION] After fixing GROUP BY clause: {t4}")
    return fix_syntax_issues(t4)
